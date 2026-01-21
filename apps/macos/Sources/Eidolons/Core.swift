//
//  Core.swift
//  Eidolons
//
//  Shell bridge for Crux core - handles event/effect loop
//

import EidolonsCore
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
    var remaining = [UInt8](requestBytes)
    while !remaining.isEmpty {
      // Deserialize the next request
      let deserializer = BincodeDeserializer(input: remaining)
      guard let request = try? SharedTypes.Request.deserialize(deserializer: deserializer) else {
        break
      }
      remaining = Array(remaining.dropFirst(deserializer.get_buffer_offset()))
      handleEffect(request: request)
    }
  }

  private func handleEffect(request: SharedTypes.Request) {
    switch request.effect {
    case .render:
      // Render effect: update view model
      let viewBytes = EidolonsShared.view()
      viewModel = try! SharedTypes.ViewModel.bincodeDeserialize(input: [UInt8](viewBytes))

    case .eidolons(let eidolonsRequest):
      // Eidolons capability: call the library and send response back
      let greeting = EidolonsCore.hello(name: eidolonsRequest.name)
      let response = SharedTypes.EidolonsResponse(greeting: greeting)
      let responseBytes = try! response.bincodeSerialize()
      let moreRequestBytes = EidolonsShared.handleResponse(
        id: request.id, data: Data(responseBytes))
      processRequests(moreRequestBytes)
    }
  }
}
