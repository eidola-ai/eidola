// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let swiftSettings: [SwiftSetting] = [
  .enableUpcomingFeature("MemberImportVisibility")
]

let binaryTarget: Target
let useLocalFramework = true

if useLocalFramework {
  binaryTarget = .binaryTarget(
    name: "EidolonsCoreRS",
    path: "target/apple/libeidolons-rs.xcframework"
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
    .iOS(.v26),
    .macOS(.v26),
  ],
  products: [
    .library(
      name: "EidolonsCore",
      targets: ["EidolonsCore"]
    )
  ],
  dependencies: [
    .package(url: "https://github.com/swiftlang/swift-testing.git", from: "6.2.3")
  ],
  targets: [
    // The built XCFramework containing the Rust library
    binaryTarget,

    // C module exposing the FFI types from the header
    .target(
      name: "eidolonsFFI",
      dependencies: [.target(name: "EidolonsCoreRS")],
      path: "swift/Sources/EidolonsCoreFFI",
      publicHeadersPath: ".",
      swiftSettings: swiftSettings
    ),

    // Swift bindings that use the FFI types
    .target(
      name: "EidolonsCore",
      dependencies: [
        .target(name: "eidolonsFFI"),
        .target(name: "EidolonsCoreRS"),
      ],
      path: "swift/Sources/EidolonsCore",
      swiftSettings: swiftSettings
    ),

    // Tests for the Swift bindings
    .testTarget(
      name: "EidolonsCoreTests",
      dependencies: [
        "EidolonsCore",
        .product(name: "Testing", package: "swift-testing"),
      ],
      path: "swift/Tests/EidolonsCoreTests",
      swiftSettings: swiftSettings
    ),
  ]
)
