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
    .package(path: "../../crates/eidola-app-core"),
  ],
  targets: [
    .target(
      name: "EidolaApp",
      dependencies: [
        .product(name: "EidolaAppCore", package: "eidola-app-core")
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
      exclude: ["Assets.xcassets"],
      swiftSettings: swiftSettings,
      linkerSettings: [
        .linkedFramework("SystemConfiguration")
      ]
    ),
    .testTarget(
      name: "EidolaTests",
      dependencies: [
        "EidolaApp",
      ],
      path: "Tests/EidolaTests",
      swiftSettings: swiftSettings
    ),
  ]
)
