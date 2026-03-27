# Eidola macOS

The reproducibility story for Apple targets is complicated. Xcode and its toolchain are actively hostile towards this goal, making publicly verifiable, reproducible iOS builds legally dubious and practically untenable. For macOS targets, Apple's continued investment into open source Swift and less restrictive policies for non-Xcode toolchain components make this more achievable. While it is possible to build and package macOS apps without Xcode, as of writing, Xcode treats all SwiftPM projects as "build only".

## Structure

The app is split into two targets to support both build systems:

- **EidolaApp** (`Sources/Eidola/`) - A library containing the SwiftUI views and app logic
- **EidolaEntrypoint** (`Sources/EidolaEntrypoint/`) - The executable that calls `EidolaAppMain.main()`

This separation allows Xcode to link against `EidolaApp` as a package dependency while maintaining `Package.swift` as the source of truth.

## Command Line (Open Source Swift)

```bash
# Build
swift build

# Build and package as .app bundle
./Support/package-app.sh release
```

The `Package.swift` is the source of truth and is (hypothetically) able to be built hermetically. While [our experiments](https://github.com/mike-marcacci/reproducibility-experiments) have demonstrated success with Swift 5.x, [we need Swift 6 to be vendored](https://github.com/NixOS/nixpkgs/issues/343210) before this will work correctly.

## Xcode

A "wrapper" Xcode project in `Xcode/` is necessary to leverage the development and debugging features of Xcode. It references `Package.swift` as a local package dependency.

XCTest is part of the Xcode toolchain and is the only realistic option for UI testing. These are therefore also part of the Xcode project.

## Support Files

The `Support/` directory contains files shared between both build systems:

- **Info.plist** - App metadata (bundle identifier, version, etc.)
- **package-app.sh** - Script to create a `.app` bundle from `swift build` output
