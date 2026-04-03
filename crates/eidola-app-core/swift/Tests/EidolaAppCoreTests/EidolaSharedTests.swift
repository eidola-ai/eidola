import Foundation
import Testing

@testable import EidolaAppCore

@Suite struct EidolaAppCoreTests {
  @Test func greetReturnsHelloWorld() {
    let greeting = greet()
    #expect(greeting == "Hello, World!")
  }
}
