//! Embed-marker recognition lockstep — the corpus proof that the two sides
//! of the quoted-references seam agree on **which markers are structural**.
//!
//! The editor (`gpui-markdown-editor::embed::embed_blocks`) recognizes an
//! `{{ embed N }}` marker via its real markdown parser: a mapped marker that
//! is the sole content of a top-level paragraph. App-core's upstream quote
//! expansion cannot depend on the editor (gpui) and instead uses
//! `eidola_common::embed::embed_marker_spans` — a zero-dependency structural
//! line scanner. If the two ever disagree, the UI and the wire disagree: a
//! marker rendered literal on screen (e.g. defused inside a code fence)
//! would silently expand upstream, or a rendered quote block would go
//! upstream as an opaque marker.
//!
//! This crate is the one place that sees both implementations, so the proof
//! lives here: for every document in the corpus, the set of `(marker text,
//! ordinal)` pairs each side recognizes must be identical. Extending either
//! implementation means extending this corpus with the cases that motivated
//! the change.

use eidola_common::embed::embed_marker_spans;
use gpui_markdown_editor::EmbedMap;
use gpui_markdown_editor::embed::embed_blocks;

/// A map covering every ordinal the corpus uses, so the editor's mapped-only
/// recognition doesn't filter differently from the (map-agnostic) scanner.
fn full_map() -> EmbedMap {
    EmbedMap::new((0..32u64).map(|n| (n, format!("content {n}"))))
}

/// Both recognizers, normalized to `(trimmed marker text, ordinal)` pairs in
/// document order. The editor reports paragraph content ranges and the
/// scanner reports line ranges (leading indent included), so compare the
/// whitespace-trimmed marker text rather than raw offsets.
fn editor_side(doc: &str) -> Vec<(String, u64)> {
    embed_blocks(doc, &full_map())
        .into_iter()
        .map(|b| {
            (
                doc[b.range].trim_matches([' ', '\t']).to_string(),
                b.ordinal,
            )
        })
        .collect()
}

fn scanner_side(doc: &str) -> Vec<(String, u64)> {
    embed_marker_spans(doc)
        .into_iter()
        .map(|s| {
            (
                doc[s.start..s.end].trim_matches([' ', '\t']).to_string(),
                s.ordinal,
            )
        })
        .collect()
}

#[test]
fn editor_and_common_scanner_recognize_the_same_markers() {
    let corpus: &[&str] = &[
        // Plain structural forms.
        "{{ embed 1 }}",
        "{{ embed 1 }}\n",
        "a\n\n{{ embed 1 }}\n\nb",
        "{{ embed 1 }}\n\n{{ embed 2 }}",
        "a\n\n{{ embed 1 }}\n\n{{ embed 2 }}\n\nb",
        "a\n\n\n\n{{ embed 3 }}\n\n\n\nb",
        "  {{ embed 4 }}",
        "a\n\n   {{ embed 5 }}\n\nb",
        "{{embed 6}}",
        "{{\tembed\t7\t}}",
        "{{ embed 007 }}",
        // Inline / same-paragraph adjacency — literal on both sides.
        "see {{ embed 1 }} here",
        "text\n{{ embed 1 }}",
        "{{ embed 1 }}\ntext",
        "x {{ embed 1 }}",
        "{{ embed 1 }} tail",
        // Containers — literal.
        "> {{ embed 1 }}",
        "- {{ embed 1 }}",
        "1. {{ embed 1 }}",
        "> quote\n>\n> {{ embed 1 }}",
        // Escapes and malformed — literal.
        "\\{{ embed 1 }}",
        "{ embed 1 }",
        "{{ embed }}",
        "{{ embed -1 }}",
        "{{ embed 1 }",
        // Indented markers: the editor's parser has indented code blocks
        // DISABLED, so these are ordinary paragraphs and DO promote — the
        // scanner must agree (this very corpus caught the divergence).
        "a\n\n    {{ embed 1 }}",
        "a\n\n\t{{ embed 1 }}",
        // Fenced code, incl. blank-line-delimited interiors (the case that
        // motivated the structural expansion rule) and unterminated fences.
        "```\n{{ embed 1 }}\n```",
        "```\n\n{{ embed 1 }}\n\n```",
        "~~~\n\n{{ embed 1 }}\n\n~~~",
        "```\n\n{{ embed 1 }}",
        "````\n```\n\n{{ embed 1 }}\n\n````",
        "```rust\n\n{{ embed 1 }}\n\n```",
        "```\ncode\n```\n\n{{ embed 1 }}",
        "a\n\n```\n\n{{ embed 1 }}\n\n```\n\n{{ embed 2 }}",
        // Block-level display math regions.
        "$$\n{{ embed 1 }}\n$$",
        "$$\n\n{{ embed 1 }}\n\n$$",
        "$$\nx^2\n$$\n\n{{ embed 1 }}",
        // Setext-heading shape.
        "{{ embed 1 }}\n====",
        // Headings / thematic breaks as neighbors (blank-line separated).
        "# title\n\n{{ embed 1 }}\n\n---\n\n{{ embed 2 }}",
        // Marker at document edges with trailing whitespace lines.
        "{{ embed 1 }}\n\n",
        "\n\n{{ embed 1 }}",
    ];

    for doc in corpus {
        let editor = editor_side(doc);
        let scanner = scanner_side(doc);
        assert_eq!(
            editor, scanner,
            "editor vs eidola-common recognition diverged on {doc:?}"
        );
    }
}
