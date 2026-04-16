import AppKit
import MarkdownEditor
import SwiftUI

@main
struct DemoApp: App {
  @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

  @State private var editorState = EditorState(
    markdown: """
      # Welcome to MarkdownEditor

      This is a simple markdown editor with WYSIWYG heading support.

      ## Getting Started

      Type some text here. Headings use the # prefix.

      ### Features

      Plain text editing and heading rendering are supported so far.
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
