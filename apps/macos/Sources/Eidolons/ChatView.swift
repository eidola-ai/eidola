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

            // Show streaming response while generating
            if core.viewModel.is_processing && !core.viewModel.streaming_text.isEmpty {
              StreamingMessageBubble(text: core.viewModel.streaming_text)
                .id("streaming")
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
        .onChange(of: core.viewModel.streaming_text) { _, _ in
          // Scroll to streaming message as it updates
          if core.viewModel.is_processing {
            withAnimation {
              proxy.scrollTo("streaming", anchor: .bottom)
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
    // Use streaming for real-time token-by-token display
    core.update(event: .submitMessageStreaming(message))
  }
}

/// Message bubble for streaming responses (assistant style with typing indicator)
struct StreamingMessageBubble: View {
  let text: String

  var body: some View {
    HStack {
      VStack(alignment: .leading, spacing: 4) {
        Text(text)
          .padding(.horizontal, 14)
          .padding(.vertical, 10)
          .background(Color(.windowBackgroundColor).opacity(0.8))
          .foregroundColor(.primary)
          .cornerRadius(16)
          .textSelection(.enabled)

        // Typing indicator
        HStack(spacing: 4) {
          ForEach(0..<3, id: \.self) { index in
            Circle()
              .fill(Color.secondary.opacity(0.6))
              .frame(width: 6, height: 6)
          }
        }
        .padding(.leading, 14)
      }

      Spacer(minLength: 60)
    }
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
