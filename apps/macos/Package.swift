// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let swiftSettings: [SwiftSetting] = [
  .enableUpcomingFeature("MemberImportVisibility")
]

let package = Package(
  name: "EidolonsApp",
  defaultLocalization: "en",
  platforms: [
    .macOS(.v26)
  ],
  products: [
    .library(
      name: "EidolonsApp",
      targets: ["EidolonsApp"]
    )
  ],
  dependencies: [
    .package(url: "https://github.com/swiftlang/swift-testing.git", from: "6.2.3"),
    .package(path: "../../eidolons"),
  ],
  targets: [
    .target(
      name: "EidolonsApp",
      dependencies: [
        .product(name: "EidolonsCore", package: "eidolons")
      ],
      path: "Sources/Eidolons",
      swiftSettings: swiftSettings
    ),
    .executableTarget(
      name: "EidolonsEntrypoint",
      dependencies: [
        "EidolonsApp"
      ],
      path: "Sources/EidolonsEntrypoint",
      resources: [
        .process("Assets.xcassets")
      ],
      swiftSettings: swiftSettings
    ),
    .testTarget(
      name: "EidolonsTests",
      dependencies: [
        "EidolonsApp",
        .product(name: "Testing", package: "swift-testing"),
      ],
      path: "Tests/EidolonsTests",
      swiftSettings: swiftSettings
    ),
  ]
)
