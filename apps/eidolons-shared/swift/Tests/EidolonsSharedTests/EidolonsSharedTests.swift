import Testing

@testable import EidolonsShared
@testable import SharedTypes

@Suite struct EidolonsSharedTests {
  @Test func processEventReturnsData() {
    // Create a simple Greet event and serialize it
    let event = Event.greet("World")
    let eventBytes = try! event.bincodeSerialize()

    // Process through the Crux core
    let responseBytes = processEvent(data: eventBytes)

    // Should return serialized requests
    #expect(!responseBytes.isEmpty)
  }
}
