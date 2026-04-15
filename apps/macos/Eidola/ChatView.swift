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
    HSplitView {
      // Space list sidebar
      VStack(alignment: .leading, spacing: 0) {
        Button("New Chat", systemImage: "plus") {
          core.newChat()
        }
        .buttonStyle(.borderless)
        .padding(8)

        Divider()

        List(
          selection: Binding(
            get: { core.currentSpaceId },
            set: { id in
              if let id {
                Task { await core.selectSpace(id) }
              }
            }
          )
        ) {
          ForEach(core.spaces, id: \.id) { space in
            Text(space.title ?? "Untitled")
              .tag(space.id)
              .lineLimit(1)
              .contextMenu {
                Button("Archive", role: .destructive) {
                  Task { await core.archiveSpace(space.id) }
                }
              }
          }
        }
        .listStyle(.sidebar)
      }
      .frame(minWidth: 160, idealWidth: 200, maxWidth: 260)

      // Chat area
      VStack(spacing: 0) {
        // Message list
        ScrollViewReader { proxy in
          ScrollView {
            LazyVStack(alignment: .leading, spacing: 12) {
              ForEach(
                Array(core.spaceMessages.enumerated()),
                id: \.offset
              ) { index, msg in
                MessageBubble(message: msg)
                  .id(index)
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
          .onChange(of: core.spaceMessages.count) {
            if !core.spaceMessages.isEmpty {
              withAnimation {
                proxy.scrollTo(core.spaceMessages.count - 1, anchor: .bottom)
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
          .disabled(
            prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || core.isLoading
          )
          .buttonStyle(.borderless)
        }
        .padding()
      }
    }
    .task {
      await core.fetchModels()
      await core.fetchSpaces()
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

#if DEBUG
  #Preview {
    ChatView(core: Core())
      .frame(width: 600, height: 600)
  }
#endif

struct MessageBubble: View {
  let message: SpaceMessage

  var body: some View {
    HStack {
      if message.role == "user" { Spacer(minLength: 60) }

      VStack(alignment: message.role == "user" ? .trailing : .leading, spacing: 4) {
        Text(message.content)
          .textSelection(.enabled)
          .padding(10)
          .background(backgroundColor)
          .foregroundStyle(foregroundColor)
          .clipShape(RoundedRectangle(cornerRadius: 12))
      }

      if message.role != "user" { Spacer(minLength: 60) }
    }
  }

  private var backgroundColor: Color {
    switch message.role {
    case "user": .accentColor
    case "error": .red.opacity(0.15)
    default: Color(.controlBackgroundColor)
    }
  }

  private var foregroundColor: Color {
    switch message.role {
    case "user": .white
    case "error": .red
    default: .primary
    }
  }
}
