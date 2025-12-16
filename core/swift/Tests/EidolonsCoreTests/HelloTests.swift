import XCTest
@testable import EidolonsCore

final class HelloTests: XCTestCase {
    func testHelloReturnsGreeting() {
        let result = hello(name: "World")
        XCTAssertEqual(result, "Hello, World!")
    }

    func testHelloWithEmptyName() {
        let result = hello(name: "")
        XCTAssertEqual(result, "Hello, !")
    }

    func testHelloWithSpecialCharacters() {
        let result = hello(name: "Rust 🦀")
        XCTAssertEqual(result, "Hello, Rust 🦀!")
    }
}
