import AppKit
import MarkdownEditor
import SwiftUI

@main
struct DemoApp: App {
  @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

  @State private var markdown = """
    # Welcome to MarkdownEditor

    This is a **bold** statement, and this is *italic* text.

    You can also do ***bold and italic*** together.

    ## Code

    Here's some `inline code` in a sentence.

    ```swift
    let greeting = "Hello, world!"
    print(greeting)
    ```

    ## Lists

    - First item
    - Second item
    - Third item

    1. Ordered one
    2. Ordered two
    3. Ordered three

    ## Other Elements

    > This is a blockquote. It should appear
    > indented with a visual indicator.

    Here's a [link to Apple](https://apple.com) in a sentence.

    This text has ~~strikethrough~~ in it.

    ---

    That was a thematic break above.
    """

  var body: some Scene {
    WindowGroup {
      HSplitView {
        MarkdownEditor(text: $markdown)
          .frame(minWidth: 400)

        ScrollView {
          Text(markdown)
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
