//
//  Core.swift
//  Eidola
//
//  Shell bridge for app core — direct UniFFI calls
//

import EidolaAppCore
import Foundation

@Observable
@MainActor
public final class Core {
  public private(set) var greeting: String

  public init() {
    greeting = ""
  }

  public func greet() {
    greeting = EidolaAppCore.greet()
  }
}
