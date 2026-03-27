// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let swiftSettings: [SwiftSetting] = [
  .enableUpcomingFeature("MemberImportVisibility")
]

let package = Package(
  name: "EidolaApp",
  defaultLocalization: "en",
  platforms: [
    .macOS(.v26)
  ],
  products: [
    .library(
      name: "EidolaApp",
      targets: ["EidolaApp"]
    )
  ],
  dependencies: [
    .package(url: "https://github.com/swiftlang/swift-testing.git", from: "6.2.3"),
    .package(path: "../../crates/eidola-shared"),
  ],
  targets: [
    .target(
      name: "EidolaApp",
      dependencies: [
        .product(name: "EidolaShared", package: "eidola-shared"),
        .product(name: "SharedTypes", package: "eidola-shared"),
        .product(name: "Serde", package: "eidola-shared"),
      ],
      path: "Sources/Eidola",
      swiftSettings: swiftSettings
    ),
    .executableTarget(
      name: "EidolaEntrypoint",
      dependencies: [
        "EidolaApp"
      ],
      path: "Sources/EidolaEntrypoint",
      resources: [
        .process("Assets.xcassets")
      ],
      swiftSettings: swiftSettings,
      linkerSettings: [
        .linkedFramework("SystemConfiguration")
      ]
    ),
    .testTarget(
      name: "EidolaTests",
      dependencies: [
        "EidolaApp",
        .product(name: "Testing", package: "swift-testing"),
      ],
      path: "Tests/EidolaTests",
      swiftSettings: swiftSettings
    ),
  ]
)
