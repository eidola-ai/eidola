// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let swiftSettings: [SwiftSetting] = [
  .enableUpcomingFeature("MemberImportVisibility")
]

let package = Package(
  name: "Eidola",
  defaultLocalization: "en",
  platforms: [
    .macOS(.v26)
  ],
  dependencies: [
    .package(path: "../../crates/eidola-app-core"),
  ],
  targets: [
    .executableTarget(
      name: "Eidola",
      dependencies: [
        .product(name: "EidolaAppCore", package: "eidola-app-core")
      ],
      path: "Eidola",
      exclude: ["Assets.xcassets"],
      swiftSettings: swiftSettings,
      linkerSettings: [
        .linkedFramework("SystemConfiguration")
      ]
    ),
    .testTarget(
      name: "EidolaTests",
      dependencies: [
        "Eidola",
      ],
      path: "EidolaTests",
      swiftSettings: swiftSettings
    ),
  ]
)
