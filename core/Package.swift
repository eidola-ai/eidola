// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let binaryTarget: Target
let useLocalFramework = true

if useLocalFramework {
  binaryTarget = .binaryTarget(
    name: "EidolonsCoreRS",
    path: "../target/apple/libeidolons-rs.xcframework"
  )
} else {
  let releaseTag = "0.1.0"
  let releaseChecksum = "67c84b83e11774ead7e78706bd73a079b49c3020ad2c866429f9644b558691fc"
  binaryTarget = .binaryTarget(
    name: "EidolonsCoreRS",
    url:
      "https://github.com/eidolons-ai/eidolons/releases/download/\(releaseTag)/libeidolons-rs.xcframework.zip",
    checksum: releaseChecksum
  )
}

let package = Package(
  name: "EidolonsCore",
  defaultLocalization: "en",
  platforms: [
    .iOS(.v16),
    .macOS(.v14),
  ],
  products: [
    .library(
      name: "EidolonsCore",
      targets: ["EidolonsCore"]
    )
  ],
  targets: [
    // The built XCFramework containing the Rust library
    binaryTarget,

    // C module exposing the FFI types from the header
    .target(
      name: "eidolonsFFI",
      dependencies: [.target(name: "EidolonsCoreRS")],
      path: "swift/Sources/EidolonsCoreFFI",
      publicHeadersPath: "."
    ),

    // Swift bindings that use the FFI types
    .target(
      name: "EidolonsCore",
      dependencies: [
        .target(name: "eidolonsFFI"),
        .target(name: "EidolonsCoreRS"),
      ],
      path: "swift/Sources/EidolonsCore"
    ),

    // Tests for the Swift bindings
    .testTarget(
      name: "EidolonsCoreTests",
      dependencies: ["EidolonsCore"],
      path: "swift/Tests/EidolonsCoreTests"
    ),
  ]
)
