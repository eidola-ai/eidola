# Mobile shells — Agent Development Guide

The native apps the App Store and Play Store distribute. Each is a thin shell around the system webview that hosts the app bundle (the wasm build of the Rust app). The shells are treated as **part of the platform**: users cannot verify the iOS one at all, so both stay small, dumb, dependency-free and readable in full. Everything that decides anything lives in the bundle.

| Path | What |
|---|---|
| `shell-contract.md` | **The contract**: every bridge call, the origin and serving rules, file semantics and durability, the bundle store and provisional activation, the digest, errors, versioning. Both shells and the app implement exactly it. |
| `ios/` | Swift, UIKit + WKWebView + Network.framework + CryptoKit. A hand-written Xcode project. |
| `android/` | Kotlin, framework APIs only (WebView, `org.json`, `java.net`), built without Gradle by the SDK's own tools. |
| `contract-test/` | A plain-JS page that exercises the whole contract against the shell it runs in, and the default embedded bundle. `shell.js` is the reference bridge. |
| `tools/` | `echo_server.py` (the test page's TCP peer), `bundle_digest.py` (the digest a shell shows, computed from a directory). |
| `device-checklist.md` | What to prepare before running a build on a physical device. |

## Doctrine

- **Six jobs and nothing else**: serve the bundle on the shell's origin; a raw TCP pipe; a flat file directory; a bundle store with provisional activation; the bundle sheet (the serving bundle's digest in native UI); fitting the webview to the screen and keyboard. A seventh is a design change: raise it, don't add it.
- **No network of its own, no trust decisions.** The shell fetches nothing, follows no link out, compares no hash, and serves only its origin. Both webviews are configured to make no requests of their own (Safe Browsing and fraud warnings off, Android metrics opted out).
- **No third-party code.** System frameworks only. Android carries the Kotlin standard library, shrunk by R8 to what the shell calls; nothing else.
- **The contract changes additively.** A new call, parameter, result member or error code bumps the contract version in `shell-contract.md`, both shells (`contractVersion` / `CONTRACT_VERSION`, the method lists), and `contract-test/test.js`; nothing existing changes meaning. A change that is not additive is a new contract and needs the app's host bridge to handle both.
- **Both shells move together.** A behaviour lands in iOS and Android in the same change, with a test-page check, and both test runs pass.
- **Nothing durable in webview storage.** iOS uses a non-persistent website data store; Android turns DOM storage off. Durable state goes through the file API.

## Building and testing

| Recipe | Script | Does |
|---|---|---|
| `just mobile-ios-sim` | `ios/build.sh` | Builds the iOS shell for the Simulator, unsigned, embedding `--bundle DIR` (default the test page); `--release`, `--device` (signed by the team in the project). |
| `just mobile-ios-test` | `ios/test-sim.sh` | XCUITest on the simulator `eidola-shell` (created once; never another device): the test page's verdict, the bundle sheet opened from the home-screen quick action and compared with the page's digest, a cold relaunch. Screenshots to `ios/build/shots`. |
| `just mobile-android` | `android/build.sh` | Builds the APK (`--bundle DIR`, `--debug`, `--out DIR`); signs it too when `KEYSTORE` is set. |
| `just mobile-android-test` | `android/test-emulator.sh` | The same run on the AVD `eidola-shell` (created once from the android-36.1 Play image; never another AVD or a device), signed with a debug key generated into `android/.keys/` (git-ignored). The sheet is opened by the launcher shortcut's intent. |
| `just mobile-android-repro` | `android/repro.sh` | Two clean release builds, identical; a signed APK strips back to them; the hash equals `android/reproducible.sha256`. |

Both test runs start `tools/echo_server.py` on `127.0.0.1:7777` (the emulator reaches it as `10.0.2.2`). A full run takes about two minutes on either platform; the page phases through several document loads, so the bundle-store checks run against real reloads.

Durability is checked through a test hook outside the contract: every flush either shell makes goes through one function (`Durable` in `Files.swift` and `Files.kt`), which in debug builds records it, and `debug.syncLog` returns and clears that record. The test page holds each file and bundle request to the contract's durability table with it; a release build answers `unknown_method` and the page skips those checks. Reverting any one of the flushes fails its check on both platforms.

The iOS check is a **local gate** (no macOS runner here): run `just mobile-ios-test` before any change to `ios/`, `contract-test/` or the contract.

## iOS

- `EidolaShell.xcodeproj` is hand-written and lists its five Swift files explicitly; a new file goes into `project.pbxproj` by hand (a build phase entry, a file reference, a group child). The embedded bundle is the folder reference `build/bundle`, which `build.sh` fills, so build through the script.
- The bundle id is `ai.eidola.mobile.dev`, the team `7N2X49G865`, iOS 26 and later.
- A document load is a main-frame navigation the navigation delegate allows; a fragment-only navigation is not one. The scheme handler then serves from the bundle selected for it.
- A content rule list blocks every load outside the `eidola` scheme (and `data:`/`blob:`), so another origin is refused by the shell and not only by the CSP.
- Durable means `F_FULLFSYNC` (plain `fsync` does not reach stable storage on Apple platforms) plus a directory `fsync`. Files are `completeUntilFirstUserAuthentication` and excluded from backup; the reasons are in the contract.
- Debug builds forward the page's console to the system log (subsystem `ai.eidola.shell`) and make the webview inspectable; Release builds do neither.
- **The iOS shell cannot be reproducibly verified.** Apple re-signs and encrypts what the App Store distributes, so nobody can compare an installed binary with a build of this source. Its smallness is the mitigation: what it may do is this document and the contract, and it is short enough to read.

## Android

- The build chain is `aapt2 link` (manifest only) → `kotlinc` against `android.jar` alone → R8 (our classes plus `kotlin-stdlib.jar`, shrunk, `keep.pro`) → `tools/ApkZip.java` (sorted, stored, one fixed timestamp) → `zipalign` → `apksigner` (v2 + v3, alignment preserved).
- `--debug` makes the APK debuggable (console in logcat, inspectable WebView); otherwise the variants are identical.
- Edge to edge, the WebView reports no safe areas and does not resize for the keyboard, so `MainActivity` pads its container by the system bars, the cutout and the IME.
- The package is `ai.eidola.mobile.dev` for now; the store package name is still to be chosen, and it cannot change once published.

### Reproducibility

The unsigned APK is a function of the source, the bundle and these pinned inputs, fetched once by `tools.sh` into `~/.cache/eidola/android-tools` and checked by SHA-256 before use (Google's repository manifest lists their SHA-1s, which match too). The local SDK is not used to build.

| Input | Version | SHA-256 of the archive |
|---|---|---|
| build-tools (macOS) | 36.1.0 | `cc56be1d4ea95041f32f4ebf3a4f5c6b990c7749d706852395006b5654eb82ff` |
| build-tools (Linux x86_64) | 36.1.0 | `a7b5889e4a79fcf3b0976bef40d401f4240fb1eed891d9d91169da1111e11d78` |
| platform `android-36` | r02 | `37607369a28c5b640b3a7998868d45898ebcb777565a0e85f9acf36f29631d2e` |
| `kotlin-compiler` | 2.3.10 | `c8d546f9ff433b529fb0ad43feceb39831040cae2ca8d17e7df46364368c9a9e` |

The JDK is not an input: kotlinc compiles against `android.jar` only (`-no-jdk`), and R8 and the zip writer produce the same bytes on any JDK from 17 up. The same source gives the same APK on macOS (arm64, JetBrains Runtime 25) and on Linux (x86_64, Temurin 17 and 21), so `reproducible.sha256` holds on both.

A **signed** APK differs on every signing (ECDSA signatures are randomized). `tools/strip_apk_signature.py SIGNED OUT` removes the v2/v3 signing block and the padding apksigner adds before it, restoring the unsigned bytes exactly; anyone can compare a published APK with their own build that way. This holds for an APK **we** publish. Play App Signing rebuilds what Play delivers (its own signature, a stamp, manifest additions), so a Play install compares by content, file by file, with Play's additions excluded.

## CI

`.github/workflows/mobile-shells.yml` runs on changes under `mobile/` (and to itself) on the `mobile` branch: it builds the Android shell twice on Linux with the pinned tools, checks the strip round trip, and checks `reproducible.sha256`. iOS is the local gate above.
