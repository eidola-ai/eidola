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
    .testTarget(
      name: "MarkdownEditorTests",
      dependencies: ["MarkdownEditor"]
    ),
  ]
)
