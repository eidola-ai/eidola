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

    case .hello(let helloRequest):
      // Hello capability: call the library and send response back
      let greeting = EidolonsShared.hello(name: helloRequest.name)
      let response = SharedTypes.HelloResponse(greeting: greeting)
      let responseBytes = try! response.bincodeSerialize()
      let moreRequestBytes = EidolonsShared.handleResponse(
        id: request.id, data: Data(responseBytes))
      processRequests(moreRequestBytes)

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
}
