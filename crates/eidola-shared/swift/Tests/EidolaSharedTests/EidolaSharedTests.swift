import Foundation
import Testing

@testable import EidolaShared
@testable import SharedTypes

@Suite struct EidolaSharedTests {
  @Test func processEventReturnsData() {
    // Create a simple Greet event and serialize it
    let event = Event.greet
    let eventBytes = try! event.bincodeSerialize()

    // Process through the Crux core
    let responseBytes = processEvent(data: Data(eventBytes))

    // Should return serialized requests
    #expect(!responseBytes.isEmpty)
  }
}
