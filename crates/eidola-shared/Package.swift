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
    name: "EidolaSharedRS",
    path: "target/apple/libeidola_shared-rs.xcframework"
  )
} else {
  let releaseTag = "0.1.0"
  let releaseChecksum = ""  // TODO: Update when releasing
  binaryTarget = .binaryTarget(
    name: "EidolaSharedRS",
    url:
      "https://github.com/eidola-ai/eidola/releases/download/\(releaseTag)/libeidola_shared-rs.xcframework.zip",
    checksum: releaseChecksum
  )
}

let package = Package(
  name: "EidolaShared",
  defaultLocalization: "en",
  platforms: [
    .macOS(.v26)
  ],
  products: [
    // The main shared core library (UniFFI bindings)
    .library(
      name: "EidolaShared",
      targets: ["EidolaShared"]
    ),
    // Crux-generated types for Event, Effect, ViewModel, etc.
    .library(
      name: "SharedTypes",
      targets: ["SharedTypes"]
    ),
    // Serde library for bincode serialization
    .library(
      name: "Serde",
      targets: ["Serde"]
    ),
  ],
  dependencies: [
    .package(url: "https://github.com/swiftlang/swift-testing.git", from: "6.2.3")
  ],
  targets: [
    // The built XCFramework containing the Rust library
    binaryTarget,

    // C module exposing the FFI types from the header
    .target(
      name: "eidola_sharedFFI",
      dependencies: [.target(name: "EidolaSharedRS")],
      path: "swift/Sources/EidolaSharedFFI",
      publicHeadersPath: ".",
      swiftSettings: swiftSettings
    ),

    // Swift bindings that use the FFI types (UniFFI generated)
    .target(
      name: "EidolaShared",
      dependencies: [
        .target(name: "eidola_sharedFFI"),
        .target(name: "EidolaSharedRS"),
      ],
      path: "swift/Sources/EidolaShared",
      swiftSettings: swiftSettings
    ),

    // Serde library for bincode serialization (Crux typegen dependency)
    .target(
      name: "Serde",
      dependencies: [],
      path: "swift/generated/SharedTypes/Sources/Serde",
      swiftSettings: swiftSettings
    ),

    // Crux-generated types (Event, Effect, ViewModel, Request, etc.)
    .target(
      name: "SharedTypes",
      dependencies: [
        .target(name: "Serde")
      ],
      path: "swift/generated/SharedTypes/Sources/SharedTypes",
      swiftSettings: swiftSettings
    ),

    // Tests for the Swift bindings
    .testTarget(
      name: "EidolaSharedTests",
      dependencies: [
        "EidolaShared",
        "SharedTypes",
        .product(name: "Testing", package: "swift-testing"),
      ],
      path: "swift/Tests/EidolaSharedTests",
      swiftSettings: swiftSettings
    ),
  ]
)
