import AppKit
import MarkdownEditor
import SwiftUI

/// Sister demo to `MarkdownEditorDemo` that exercises the TextKit 2 rendering
/// path. Uses the same sample markdown so the two windows can be opened
/// side-by-side for visual comparison while the migration progresses.
///
/// Phase 1 status: only base typography, paragraph styling, and font traits
/// are applied via the TK2 path. Glyph hiding (delimiters, list markers,
/// checkboxes), code-block backgrounds, blockquote borders, and cursor-driven
/// delimiter coloring are not yet implemented — the markdown source renders
/// verbatim.
@main
struct TextKit2DemoApp: App {
  @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

  @State private var editorState = EditorState(
    markdown: """
      # Welcome to MarkdownEditor

      A WYSIWYG markdown editor where formatting is visible inline. Click on any formatted text to reveal the underlying markdown syntax.

      ## Text Formatting

      This is **bold text**, *italic text*, ***bold and italic***, ~~strikethrough~~, and `inline code`.

      ## Links & Images

      Here's a [link to Apple](https://apple.com) and an ![image alt](https://example.com/image.png) reference.

      ## Lists

      Unordered lists use bullet markers:

      - First item
      - Second item with **bold** inside
      - Third item

      Ordered lists auto-number:

      1. First ordered
      2. Second ordered
      3. Third ordered

      Task lists use checkboxes:

      - [ ] Unchecked task
      - [x] Completed task
      - [ ] Another task

      ## Code Blocks

      ```swift
      let greeting = "Hello, world!"
      print(greeting)
      ```

      ## Blockquotes

      > This is a blockquote that can span
      > multiple lines with the `>` prefix.

      ---

      That's a horizontal rule above. Try editing any of these elements!
      """)

  var body: some Scene {
    WindowGroup("MarkdownEditor — TextKit 2 (Phase 1)") {
      VStack(spacing: 0) {
        HStack {
          Text("TextKit 2 path · Phase 1")
            .font(.system(.caption, design: .rounded).weight(.semibold))
          Text(
            "base typography only — glyph hiding, custom fragments, "
              + "and rendering attributes land in later phases")
            .font(.caption)
            .foregroundStyle(.secondary)
          Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.thinMaterial)

        HSplitView {
          MarkdownEditor(state: $editorState, useTextKit2: true)
            .frame(minWidth: 400)

          ScrollView {
            Text(editorState.markdown)
              .font(.system(.body, design: .monospaced))
              .padding()
              .frame(maxWidth: .infinity, alignment: .leading)
              .textSelection(.enabled)
          }
          .frame(minWidth: 300)
          .background(.background)
        }
      }
      .frame(minWidth: 700, minHeight: 500)
    }
    .windowStyle(.titleBar)
  }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
  func applicationDidFinishLaunching(_ notification: Notification) {
    NSApp.setActivationPolicy(.regular)
    NSApp.activate()
  }
}
