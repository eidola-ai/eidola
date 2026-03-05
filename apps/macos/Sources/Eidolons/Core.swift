//
//  Core.swift
//  Eidolons
//
//  Shell bridge for Crux core - handles event/effect loop
//

import EidolonsShared
import Foundation
import Serde
import SharedTypes

/// Represents a streaming response type that is safely Sendable
enum StreamingResponseType: Sendable {
  case chunk(String)
  case done
  case error(String)

  func toSharedType() -> SharedTypes.PerceptionStreamingResponse {
    switch self {
    case .chunk(let text): return .chunk(text)
    case .done: return .done
    case .error(let error): return .error(error)
    }
  }
}

/// Callback handler for streaming perception that routes responses through Crux.
/// Each chunk is sent via handleResponse to maintain Crux's event loop.
final class StreamingResponseHandler: StreamingCallback, @unchecked Sendable {
  private let requestId: UInt32
  private let sendResponse: @Sendable (UInt32, StreamingResponseType) -> Void

  init(
    requestId: UInt32,
    sendResponse: @escaping @Sendable (UInt32, StreamingResponseType) -> Void
  ) {
    self.requestId = requestId
    self.sendResponse = sendResponse
  }

  func onChunk(text: String) {
    sendResponse(requestId, .chunk(text))
  }

  func onComplete() {
    sendResponse(requestId, .done)
  }

  func onError(error: String) {
    sendResponse(requestId, .error(error))
  }
}

@Observable
@MainActor
public final class Core {
  public private(set) var viewModel: SharedTypes.ViewModel

  /// The perception service for AI inference
  private let perceptionService: PerceptionService

  public init() {
    // Initialize perception service (model loading happens lazily via initialize())
    perceptionService = PerceptionService()

    // Get initial view model from core
    let viewBytes = EidolonsShared.view()
    viewModel = try! SharedTypes.ViewModel.bincodeDeserialize(input: [UInt8](viewBytes))
  }

  public func update(event: SharedTypes.Event) {
    let eventBytes = try! event.bincodeSerialize()
    let requestBytes = EidolonsShared.processEvent(data: Data(eventBytes))
    processRequests(requestBytes)
  }

  private func processRequests(_ requestBytes: Data) {
    guard !requestBytes.isEmpty else { return }

    let deserializer = BincodeDeserializer(input: [UInt8](requestBytes))

    // The response is a bincode-serialized Vec<Request>, so read the length first
    guard let count = try? deserializer.deserialize_len() else { return }

    for _ in 0..<count {
      guard let request = try? SharedTypes.Request.deserialize(deserializer: deserializer)
      else {
        break
      }
      handleEffect(request: request)
    }
  }

  private func handleEffect(request: SharedTypes.Request) {
    switch request.effect {
    case .render:
      // Render effect: update view model
      let viewBytes = EidolonsShared.view()
      viewModel = try! SharedTypes.ViewModel.bincodeDeserialize(input: [UInt8](viewBytes))

    case .perception(let perceptionRequest):
      // Perception capability: call the AI service asynchronously
      let requestId = request.id
      // Convert Crux ChatMessage to UniFFI ServiceChatMessage
      let messages = perceptionRequest.messages.map { msg in
        ServiceChatMessage(
          role: msg.role == .user ? .user : .assistant,
          content: msg.content
        )
      }
      Task {
        await handlePerception(requestId: requestId, messages: messages)
      }

    case .perceptionStreaming(let streamingRequest):
      // Streaming perception: call AI service and send responses via handleResponse
      let requestId = request.id
      let messages = streamingRequest.messages.map { msg in
        ServiceChatMessage(
          role: msg.role == .user ? .user : .assistant,
          content: msg.content
        )
      }
      Task {
        await handlePerceptionStreaming(requestId: requestId, messages: messages)
      }
    }
  }

  /// Handles perception requests asynchronously
  private func handlePerception(requestId: UInt32, messages: [ServiceChatMessage]) async {
    // Ensure the service is initialized (downloads model if needed)
    let isReady = await perceptionService.isReady()
    if !isReady {
      do {
        try await perceptionService.initialize()
      } catch {
        // On initialization failure, return an error message
        let response = SharedTypes.PerceptionResponse(
          response: "Error initializing AI: \(error.localizedDescription)")
        sendPerceptionResponse(requestId: requestId, response: response)
        return
      }
    }

    // Call the AI service with full conversation history
    do {
      let result = try await perceptionService.chat(messages: messages)
      let response = SharedTypes.PerceptionResponse(response: result)
      sendPerceptionResponse(requestId: requestId, response: response)
    } catch {
      let response = SharedTypes.PerceptionResponse(
        response: "Error: \(error.localizedDescription)")
      sendPerceptionResponse(requestId: requestId, response: response)
    }
  }

  /// Sends a perception response back to the core
  private func sendPerceptionResponse(requestId: UInt32, response: SharedTypes.PerceptionResponse) {
    let responseBytes = try! response.bincodeSerialize()
    let moreRequestBytes = EidolonsShared.handleResponse(
      id: requestId, data: Data(responseBytes))
    processRequests(moreRequestBytes)
  }

  /// Handles streaming perception requests by calling handleResponse for each chunk
  private func handlePerceptionStreaming(requestId: UInt32, messages: [ServiceChatMessage]) async {
    // Ensure the service is initialized
    let isReady = await perceptionService.isReady()
    if !isReady {
      do {
        try await perceptionService.initialize()
      } catch {
        // On initialization failure, send error response
        sendStreamingResponse(
          requestId: requestId,
          response: .error("Error initializing AI: \(error.localizedDescription)"))
        return
      }
    }

    // Create callback handler that sends responses through Crux's handleResponse
    // Note: We dispatch to MainActor because sendStreamingResponse modifies Core state
    let handler = StreamingResponseHandler(requestId: requestId) {
      [weak self] id, response in
      Task { @MainActor in
        self?.sendStreamingResponse(requestId: id, response: response.toSharedType())
      }
    }

    // Call the streaming API
    do {
      try await perceptionService.chatStreaming(messages: messages, callback: handler)
    } catch {
      sendStreamingResponse(
        requestId: requestId, response: .error("Streaming error: \(error.localizedDescription)"))
    }
  }

  /// Sends a streaming response back to the core via handleResponse
  private func sendStreamingResponse(
    requestId: UInt32, response: SharedTypes.PerceptionStreamingResponse
  ) {
    let responseBytes = try! response.bincodeSerialize()
    let moreRequestBytes = EidolonsShared.handleResponse(
      id: requestId, data: Data(responseBytes))
    processRequests(moreRequestBytes)
  }
}
