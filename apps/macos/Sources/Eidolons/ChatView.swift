//
//  ChatView.swift
//  Eidolons
//
//  Chat interface for AI conversations
//

import SharedTypes
import SwiftUI

struct ChatView: View {
  @Bindable var core: Core

  @State private var inputText: String = ""

  var body: some View {
    VStack(spacing: 0) {
      // Message list
      ScrollViewReader { proxy in
        ScrollView {
          LazyVStack(alignment: .leading, spacing: 12) {
            ForEach(Array(core.viewModel.conversation.enumerated()), id: \.offset) {
              index, message in
              MessageBubble(message: message)
                .id(index)
            }
          }
          .padding()
        }
        .onChange(of: core.viewModel.conversation.count) { _, newCount in
          // Scroll to the latest message
          if newCount > 0 {
            withAnimation {
              proxy.scrollTo(newCount - 1, anchor: .bottom)
            }
          }
        }
      }

      Divider()

      // Input area
      HStack(spacing: 12) {
        TextField("Type a message...", text: $inputText)
          .textFieldStyle(.plain)
          .padding(10)
          .background(Color(.textBackgroundColor))
          .cornerRadius(8)
          .disabled(core.viewModel.is_processing)
          .onSubmit {
            sendMessage()
          }

        if core.viewModel.is_processing {
          ProgressView()
            .controlSize(.small)
            .frame(width: 32, height: 32)
        } else {
          Button(action: sendMessage) {
            Image(systemName: "arrow.up.circle.fill")
              .font(.system(size: 28))
              .foregroundColor(inputText.isEmpty ? .gray : .accentColor)
          }
          .buttonStyle(.plain)
          .disabled(inputText.isEmpty)
        }
      }
      .padding()
    }
  }

  private func sendMessage() {
    let message = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !message.isEmpty else { return }

    inputText = ""
    core.update(event: .submitMessage(message))
  }
}

struct MessageBubble: View {
  let message: ChatMessage

  var body: some View {
    HStack {
      if message.role == .user {
        Spacer(minLength: 60)
      }

      Text(message.content)
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(bubbleColor)
        .foregroundColor(textColor)
        .cornerRadius(16)
        .textSelection(.enabled)

      if message.role == .assistant {
        Spacer(minLength: 60)
      }
    }
  }

  private var bubbleColor: Color {
    switch message.role {
    case .user:
      return .accentColor
    case .assistant:
      return Color(.windowBackgroundColor).opacity(0.8)
    }
  }

  private var textColor: Color {
    switch message.role {
    case .user:
      return .white
    case .assistant:
      return .primary
    }
  }
}

#Preview {
  ChatView(core: Core())
    .frame(width: 400, height: 600)
}
