import AppKit
import MarkdownEditor
import SwiftUI

@main
struct DemoApp: App {
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
    WindowGroup {
      HSplitView {
        MarkdownEditor(state: $editorState)
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
      .frame(minWidth: 700, minHeight: 500)
    }
    .windowStyle(.titleBar)
  }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
  func applicationDidFinishLaunching(_ notification: Notification) {
    // Ensure the app appears in the Dock and can receive keyboard focus,
    // even when launched as a bare executable outside a .app bundle.
    NSApp.setActivationPolicy(.regular)
    NSApp.activate()
  }
}
