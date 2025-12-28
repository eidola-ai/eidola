import Testing

@testable import EidolonsCore

@Suite struct HelloTests {
  @Test
  func helloReturnsGreeting() {
    let result = hello(name: "World")
    #expect(result == "Hello, World!")
  }

  @Test
  func helloWithEmptyName() {
    let result = hello(name: "")
    #expect(result == "Hello, !")
  }

  @Test
  func helloWithSpecialCharacters() {
    let result = hello(name: "Rust 🦀")
    #expect(result == "Hello, Rust 🦀!")
  }
}
