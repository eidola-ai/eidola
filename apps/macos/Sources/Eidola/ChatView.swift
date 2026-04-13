//
//  ChatView.swift
//  Eidola
//

import EidolaAppCore
import SwiftUI

struct ChatView: View {
  var core: Core
  @State private var prompt = ""
  @State private var selectedModel = "glm-5-1"
  @FocusState private var promptFocused: Bool

  var body: some View {
    VStack(spacing: 0) {
      // Message list
      ScrollViewReader { proxy in
        ScrollView {
          LazyVStack(alignment: .leading, spacing: 12) {
            ForEach(core.chatHistory) { entry in
              MessageBubble(entry: entry)
                .id(entry.id)
            }

            if core.isLoading {
              HStack {
                ProgressView()
                  .controlSize(.small)
                Text("Thinking...")
                  .foregroundStyle(.secondary)
              }
              .padding(.horizontal)
              .id("loading")
            }
          }
          .padding()
        }
        .onChange(of: core.chatHistory.count) {
          if let last = core.chatHistory.last {
            withAnimation {
              proxy.scrollTo(last.id, anchor: .bottom)
            }
          }
        }
      }

      Divider()

      // Input area
      HStack(spacing: 8) {
        Picker("Model", selection: $selectedModel) {
          Text("glm-5-1").tag("glm-5-1")
          ForEach(core.models, id: \.id) { model in
            Text(model.id).tag(model.id)
          }
        }
        .labelsHidden()
        .frame(width: 140)

        TextField("Message...", text: $prompt, axis: .vertical)
          .textFieldStyle(.plain)
          .lineLimit(1...5)
          .focused($promptFocused)
          .onSubmit { sendMessage() }

        Button(action: sendMessage) {
          Image(systemName: "arrow.up.circle.fill")
            .font(.title2)
        }
        .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || core.isLoading)
        .buttonStyle(.borderless)
      }
      .padding()
    }
    .navigationTitle("Chat")
    .toolbar {
      ToolbarItem(placement: .automatic) {
        Button("Clear", systemImage: "trash") {
          core.clearChat()
        }
        .disabled(core.chatHistory.isEmpty)
      }
    }
    .task {
      await core.fetchModels()
    }
    .onAppear {
      promptFocused = true
    }
  }

  private func sendMessage() {
    let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty, !core.isLoading else { return }
    prompt = ""
    Task {
      await core.sendMessage(text, model: selectedModel)
    }
  }
}

struct MessageBubble: View {
  let entry: ChatEntry

  var body: some View {
    HStack {
      if entry.role == .user { Spacer(minLength: 60) }

      VStack(alignment: entry.role == .user ? .trailing : .leading, spacing: 4) {
        Text(entry.content)
          .textSelection(.enabled)
          .padding(10)
          .background(backgroundColor)
          .foregroundStyle(foregroundColor)
          .clipShape(RoundedRectangle(cornerRadius: 12))

        if let tokens = entry.inputTokens, let outTokens = entry.outputTokens {
          Text("\(tokens) in / \(outTokens) out")
            .font(.caption2)
            .foregroundStyle(.tertiary)
        }
      }

      if entry.role != .user { Spacer(minLength: 60) }
    }
  }

  private var backgroundColor: Color {
    switch entry.role {
    case .user: .accentColor
    case .assistant: Color(.controlBackgroundColor)
    case .error: .red.opacity(0.15)
    }
  }

  private var foregroundColor: Color {
    switch entry.role {
    case .user: .white
    case .assistant: .primary
    case .error: .red
    }
  }
}
