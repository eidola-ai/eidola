//
//  Core.swift
//  Eidola
//
//  Shell bridge for Crux core - handles event/effect loop
//

import EidolaShared
import Foundation
import Serde
import SharedTypes

@Observable
@MainActor
public final class Core {
  public private(set) var viewModel: SharedTypes.ViewModel

  public init() {
    let viewBytes = EidolaShared.view()
    viewModel = try! SharedTypes.ViewModel.bincodeDeserialize(input: [UInt8](viewBytes))
  }

  public func update(event: SharedTypes.Event) {
    let eventBytes = try! event.bincodeSerialize()
    let requestBytes = EidolaShared.processEvent(data: Data(eventBytes))
    processRequests(requestBytes)
  }

  private func processRequests(_ requestBytes: Data) {
    guard !requestBytes.isEmpty else { return }

    let deserializer = BincodeDeserializer(input: [UInt8](requestBytes))

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
      let viewBytes = EidolaShared.view()
      viewModel = try! SharedTypes.ViewModel.bincodeDeserialize(input: [UInt8](viewBytes))
    }
  }
}
