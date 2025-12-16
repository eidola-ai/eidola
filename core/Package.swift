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
      targets: [
        "EidolonsCoreFFI"

        // TODO: Wrap the FFI bindings in custom, idiomatic Swift.
        // "Eidolons",
      ]
    )
  ],
  targets: [
    // Target the built XCFramework.
    binaryTarget,

    // Target the generated FFI bindings, which depend on the XCFramework.
    .target(
      name: "EidolonsCoreFFI",
      dependencies: [.target(name: "EidolonsCoreRS")],
      path: "swift/Sources/EidolonsCore"
    ),

    // Tests for the FFI bindings
    .testTarget(
      name: "EidolonsCoreFFITests",
      dependencies: ["EidolonsCoreFFI"],
      path: "swift/Tests/EidolonsCoreTests"
    ),

    // TODO: Wrap the FFI bindings in custom, idiomatic Swift.
    // .target(
    //     name: "Eidolons",
    //     dependencies: [.target(name: "EidolonsCoreFFI")],
    //     path: "apple/Sources/Eidolons"
    // ),

    // TODO: Add tests for the Swift wrappers.
    // .testTarget(
    //     name: "EidolonsTests",
    //     dependencies: ["Eidolons"],
    //     path: "apple/Tests/EidolonsTests"
    // ),
  ]
)
