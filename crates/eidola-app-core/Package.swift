// swift-tools-version: 6.2

import PackageDescription

let swiftSettings: [SwiftSetting] = [
  .enableUpcomingFeature("MemberImportVisibility")
]

let binaryTarget: Target
let useLocalFramework = true

if useLocalFramework {
  binaryTarget = .binaryTarget(
    name: "EidolaAppCoreRS",
    path: "target/apple/libeidola_app_core-rs.xcframework"
  )
} else {
  let releaseTag = "0.1.0"
  let releaseChecksum = ""  // TODO: Update when releasing
  binaryTarget = .binaryTarget(
    name: "EidolaAppCoreRS",
    url:
      "https://github.com/eidola-ai/eidola/releases/download/\(releaseTag)/libeidola_app_core-rs.xcframework.zip",
    checksum: releaseChecksum
  )
}

let package = Package(
  name: "EidolaAppCore",
  defaultLocalization: "en",
  platforms: [
    .macOS(.v26)
  ],
  products: [
    .library(
      name: "EidolaAppCore",
      targets: ["EidolaAppCore"]
    )
  ],
  dependencies: [
    .package(url: "https://github.com/swiftlang/swift-testing.git", from: "6.2.3")
  ],
  targets: [
    binaryTarget,

    .target(
      name: "eidola_app_coreFFI",
      dependencies: [.target(name: "EidolaAppCoreRS")],
      path: "swift/Sources/EidolaAppCoreFFI",
      publicHeadersPath: ".",
      swiftSettings: swiftSettings
    ),

    .target(
      name: "EidolaAppCore",
      dependencies: [
        .target(name: "eidola_app_coreFFI"),
        .target(name: "EidolaAppCoreRS"),
      ],
      path: "swift/Sources/EidolaAppCore",
      swiftSettings: swiftSettings
    ),

    .testTarget(
      name: "EidolaAppCoreTests",
      dependencies: [
        "EidolaAppCore",
        .product(name: "Testing", package: "swift-testing"),
      ],
      path: "swift/Tests/EidolaAppCoreTests",
      swiftSettings: swiftSettings
    ),
  ]
)
