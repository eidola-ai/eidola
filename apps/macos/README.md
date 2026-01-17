# Eidolons macOS

The reproducibility store for Apple targets is complicated. Xcode and its toolchain are actively hostile towards this goal, making publicly verifiable, reproducible iOS builds legally dubious and practically untenable. For macOS targets, Apple's continued investment into open source Swift and less restrictive policies for non-Xcode toolchain components make this more achievable. While it is possible to build and package macOS apps without XCode, as of writing, XCode treats all SwiftPM projects as "build only".

## Command Line (Open Source Swift)

The `Package.swift` is the source of truth and is (hypothetically) able to be built hermetically. While [our experiments](https://github.com/mike-marcacci/reproducibility-experiments) have demonstrated success with Swift 5.x, [we need Swift 6 to be vendored](https://github.com/NixOS/nixpkgs/issues/343210) before this will work correctly.

## Xcode

A "wrapper" Xcode project is necessary to leverage the development and debugging features of XCode. This and its support files exist in ./Xcode.

XCTest is part of the Xcode toolchain, and is the only realistic option for UI testing. These are therefore also part of the Xcode project.
