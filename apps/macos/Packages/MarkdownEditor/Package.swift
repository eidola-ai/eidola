// swift-tools-version: 6.2

import PackageDescription

let package = Package(
  name: "MarkdownEditor",
  platforms: [
    .macOS(.v26)
  ],
  products: [
    .library(name: "MarkdownEditor", targets: ["MarkdownEditor"])
  ],
  dependencies: [
    .package(url: "https://github.com/swiftlang/swift-markdown", from: "0.7.0")
  ],
  targets: [
    .target(
      name: "MarkdownEditor",
      dependencies: [
        .product(name: "Markdown", package: "swift-markdown")
      ]
    ),
    .executableTarget(
      name: "MarkdownEditorDemo",
      dependencies: ["MarkdownEditor"]
    ),
    .executableTarget(
      name: "MarkdownEditorScript",
      dependencies: ["MarkdownEditor"]
    ),
    .executableTarget(
      name: "BlockRendererSpikeS1"
    ),
    .executableTarget(
      name: "BlockRendererSpikeS1_1"
    ),
    .executableTarget(
      name: "BlockRendererSpikeS2"
    ),
    .executableTarget(
      name: "BlockRendererSpikeS3"
    ),
    .testTarget(
      name: "MarkdownEditorTests",
      dependencies: ["MarkdownEditor"]
    ),
  ]
)
