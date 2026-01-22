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

  public init() {
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
    }
  }
}
