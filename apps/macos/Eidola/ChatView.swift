//
//  ChatView.swift
//  Eidola
//

import EidolaAppCore
import MarkdownUI
import SwiftUI

struct ChatView: View {
  var core: Core
  @State private var prompt = ""
  @State private var spaceId: String?
  @State private var messages: [SpaceMessage] = []
  @State private var isLoading = false
  @FocusState private var promptFocused: Bool

  var body: some View {
    ScrollViewReader { proxy in
      ScrollView {
        VStack(alignment: .leading, spacing: 0) {
          ForEach(
            Array(messages.enumerated()),
            id: \.offset
          ) { index, msg in
            MessageBlock(message: msg)
              .id(index)
          }

          if isLoading {
            HStack(spacing: 6) {
              ProgressView()
                .controlSize(.small)
              Text("Thinking...")
                .foregroundStyle(.tertiary)
                .font(.body)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .background(Color(.controlBackgroundColor).opacity(0.4))
            .id("loading")
          }

          // Inline composition area
          TextEditor(text: $prompt)
            .font(.body)
            .scrollDisabled(true)
            .frame(minHeight: 40)
            .fixedSize(horizontal: false, vertical: true)
            .scrollContentBackground(.hidden)
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .focused($promptFocused)
            .id("input")
        }
      }
      .onChange(of: messages.count) {
        withAnimation {
          proxy.scrollTo("input", anchor: .bottom)
        }
      }
    }
    .background(Color(.textBackgroundColor))
    .task {
      await core.fetchModels()
    }
    .onAppear {
      promptFocused = true
    }
    .onKeyPress(phases: .down) { keyPress in
      if keyPress.key == .return && keyPress.modifiers == .command {
        requestFeedback()
        return .handled
      }
      return .ignored
    }
  }

  private func requestFeedback() {
    let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty, !isLoading else { return }
    prompt = ""

    messages.append(SpaceMessage(role: "user", content: text))
    isLoading = true

    Task {
      do {
        let result = try await core.chat(
          prompt: text, model: "glm-5-1", spaceId: spaceId)
        spaceId = result.spaceId
        messages = try await core.getSpaceMessages(spaceId: result.spaceId)
      } catch {
        messages.append(SpaceMessage(role: "error", content: error.localizedDescription))
      }
      isLoading = false
    }
  }
}

#if DEBUG
  #Preview {
    ChatView(core: Core())
      .frame(width: 600, height: 600)
  }
#endif

struct MessageBlock: View {
  let message: SpaceMessage

  var body: some View {
    Markdown(message.content)
      .textSelection(.enabled)
      .foregroundStyle(message.role == "error" ? .red : .primary)
      .frame(maxWidth: .infinity, alignment: .leading)
      .padding(.horizontal, 20)
      .padding(.vertical, 12)
      .background(backgroundColor)
  }

  private var backgroundColor: Color {
    switch message.role {
    case "user": .clear
    case "error": .red.opacity(0.06)
    default: Color(.controlBackgroundColor).opacity(0.4)
    }
  }
}
