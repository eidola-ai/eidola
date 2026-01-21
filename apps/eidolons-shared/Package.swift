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
    name: "EidolonsSharedRS",
    path: "target/apple/libeidolons_shared-rs.xcframework"
  )
} else {
  let releaseTag = "0.1.0"
  let releaseChecksum = ""  // TODO: Update when releasing
  binaryTarget = .binaryTarget(
    name: "EidolonsSharedRS",
    url:
      "https://github.com/eidolons-ai/eidolons/releases/download/\(releaseTag)/libeidolons_shared-rs.xcframework.zip",
    checksum: releaseChecksum
  )
}

let package = Package(
  name: "EidolonsShared",
  defaultLocalization: "en",
  platforms: [
    .macOS(.v26)
  ],
  products: [
    // The main shared core library (UniFFI bindings)
    .library(
      name: "EidolonsShared",
      targets: ["EidolonsShared"]
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
      name: "eidolons_sharedFFI",
      dependencies: [.target(name: "EidolonsSharedRS")],
      path: "swift/Sources/EidolonsSharedFFI",
      publicHeadersPath: ".",
      swiftSettings: swiftSettings
    ),

    // Swift bindings that use the FFI types (UniFFI generated)
    .target(
      name: "EidolonsShared",
      dependencies: [
        .target(name: "eidolons_sharedFFI"),
        .target(name: "EidolonsSharedRS"),
      ],
      path: "swift/Sources/EidolonsShared",
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
      name: "EidolonsSharedTests",
      dependencies: [
        "EidolonsShared",
        "SharedTypes",
        .product(name: "Testing", package: "swift-testing"),
      ],
      path: "swift/Tests/EidolonsSharedTests",
      swiftSettings: swiftSettings
    ),
  ]
)
