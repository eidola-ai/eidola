# Mobile shells — Agent Development Guide

The native apps the App Store and Play Store distribute. Each is a thin shell around the system webview that hosts the app bundle (the wasm build of the Rust app). The shells are treated as **part of the platform**: users cannot verify the iOS one at all, so both stay small, dumb, dependency-free and readable in full. Everything that decides anything lives in the bundle.

| Path | What |
|---|---|
| `shell-contract.md` | **The contract**: every bridge call, the origin and serving rules, file semantics and durability, the bundle store and provisional activation, the digest, errors, versioning. Both shells and the app implement exactly it. |
| `ios/` | Swift, UIKit + WKWebView + Network.framework + CryptoKit. A hand-written Xcode project. |
| `android/` | Kotlin, framework APIs only (WebView, `org.json`, `java.net`), built without Gradle by the SDK's own tools. |
| `contract-test/` | A plain-JS page that exercises the whole contract against the shell it runs in, and the default embedded bundle. `shell.js` is the reference bridge. |
| `tools/` | `echo_server.py` (the test page's TCP peer), `bundle_digest.py` (the digest a shell shows, computed from a directory), `bundle_container.py` (a bundle directory as the `ustar` container a release publishes it in, and its `artifact-manifest.json` row; the same bytes as app-core's `updater::mobile::container::write_ustar`). |
| `build-bundle.sh`, `bundle-repro.sh`, `self-test.sh` | The app bundle (`crates/eidola-mobile`, wasm32) built reproducibly, checked by two clean builds, and its headless self-test under either shell; see `crates/eidola-mobile/AGENTS.md`. |
| `build-bundle-release.sh`, `bundle-release.digest` | The release bundle: the same app built in the pinned StageX toolchain, the bundle a release records and the release APK embeds, and its digest, which CI checks on another host; see `crates/eidola-mobile/AGENTS.md` → The release bundle. |
| `device-checklist.md` | What to prepare before running a build on a physical device. |

## Doctrine

- **Seven jobs and nothing else**: serve the bundle on the shell's origin; a raw TCP pipe; a flat file directory; a bundle store with provisional activation; the bundle sheet (the serving bundle's digest in native UI); fitting the webview to the screen and keyboard; handing `https:` links to the system browser (purchases happen on the web). An eighth is a design change: raise it, don't add it.
- **No network of its own, no trust decisions.** The shell fetches nothing, loads no other origin in its webview, compares no hash, and serves only its origin; an external link goes to the system browser, never an in-app one. Both webviews are configured to make no requests of their own (Safe Browsing and fraud warnings off, Android metrics opted out).
- **No third-party code.** System frameworks only. Android carries the Kotlin standard library, shrunk by R8 to what the shell calls; nothing else.
- **The contract changes additively.** A new call, parameter, result member or error code bumps the contract version in `shell-contract.md`, both shells (`contractVersion` / `CONTRACT_VERSION`, the method lists), and `contract-test/test.js`; nothing existing changes meaning. A change that is not additive is a new contract and needs the app's host bridge to handle both.
- **Nothing a page sends can crash the shell.** Request input is parsed in one place per shell (`SchemeHandler.target` on iOS, `bundlePath` on Android), which yields a bundle path or a 4xx; no force-unwrap or unchecked slice touches it.
- **Both shells move together.** A behaviour lands in iOS and Android in the same change, with a test-page check, and both test runs pass.
- **The published trust model is `docs/mobile.md`**: what each platform lets a user verify, and what the shell is trusted with. A change to what a shell does, or to what the app checks before handing it anything, updates that page in the same change.
- **Nothing durable in webview storage.** iOS uses a non-persistent website data store; Android turns DOM storage off. Durable state goes through the file API.

## Icons

Both launcher icons are generated from `brand/` (`just update-brand`, macOS only) and committed: `ios/AppIcon.icon` (an Icon Composer document Xcode compiles) and `android/res/` (an adaptive icon whose foreground is also the themed-icon layer). Never edit them by hand; `brand/AGENTS.md` has the geometry and why each looks as it does.

## Building and testing

Both shells embed **the app bundle** by default (`build.sh` without `--bundle` runs `../build-bundle.sh` and embeds `build/bundle`); their own tests embed the contract test page (`--bundle ../contract-test`), and so does `repro.sh` without arguments, which keeps `reproducible.sha256` independent of the Rust build.

| Recipe | Script | Does |
|---|---|---|
| `just mobile-bundle` | `build-bundle.sh` | The app bundle into `build/bundle` (`--self-test`: the self-test bundle). Prints sizes and the bundle digest. |
| `just mobile-bundle-repro` | `bundle-repro.sh` | Two builds from scratch, compared file by file. |
| `just mobile-bundle-release` | `build-bundle-release.sh` | The release bundle into `build/bundle-release`, built in Docker (`--check` against `bundle-release.digest`, `--update` to rewrite it, `--no-cache`). |
| `just mobile-self-test ios\|android` | `self-test.sh` | The app's self-test bundle under a debug shell, on its own simulator or AVD (`eidola-bridge`), against shims and an echo server run here: attestation over the pipe and its refusal, the durability barrier, the profile across a cold relaunch, throughput. Evidence to `build/self-test/<platform>/`. |
| `just mobile-ios-sim` | `ios/build.sh` | Builds the iOS shell for the Simulator, unsigned, embedding `--bundle DIR` (default the test page); `--release`, `--device` (signed by the team in the project). |
| `just mobile-ios-test` | `ios/test-sim.sh` | XCUITest on the simulator `eidola-shell` (created once; never another device): the test page's verdict, the bundle sheet opened from the home-screen quick action and compared with the page's digest, a cold relaunch. Screenshots to `ios/build/shots`. |
| `just mobile-android` | `android/build.sh` | Builds the APK (`--bundle DIR`, `--debug`, `--out DIR`); signs it too when `KEYSTORE` is set; `--aab` also builds the App Bundle Play takes. |
| `just mobile-android-test` | `android/test-emulator.sh` | The same run on the AVD `eidola-shell` (created once from the android-36.1 Play image; never another AVD or a device), signed with a debug key generated into `android/.keys/` (git-ignored). The sheet is opened by the launcher shortcut's intent. |
| `just mobile-android-repro` | `android/repro.sh` | Two clean release builds, identical APKs and App Bundles; a signed APK strips back to the unsigned one; the APK bundletool generates from the App Bundle carries the same code and bundle; both hashes equal `android/reproducible.sha256`. |

Both test runs start `tools/echo_server.py` on `127.0.0.1:7777` (the emulator reaches it as `10.0.2.2`). A full run takes about two minutes on either platform; the page phases through several document loads, so the bundle-store checks run against real reloads.

Durability is checked through a test hook outside the contract: every flush either shell makes goes through one function (`Durable` in `Files.swift` and `Files.kt`), which in debug builds records it, and `debug.syncLog` returns and clears that record. The test page holds each file and bundle request to the contract's durability table with it; a release build answers `unknown_method` and the page skips those checks. Reverting any one of the flushes fails its check on both platforms.

The iOS check is a **local gate** (no macOS runner here): run `just mobile-ios-test` before any change to `ios/`, `contract-test/` or the contract.

## iOS

- `EidolaShell.xcodeproj` is hand-written and lists its five Swift files and `AppIcon.icon` explicitly; a new file goes into `project.pbxproj` by hand (a build phase entry, a file reference, a group child). The embedded bundle is the folder reference `build/bundle`, which `build.sh` fills, so build through the script.
- The bundle id is `ai.eidola.mobile` for the Release configuration (`--release`) and `ai.eidola.mobile.dev` for Debug, so a dev build installs beside a release one; the team is `7N2X49G865`, iOS 26 and later.
- A document load is a main-frame navigation the navigation delegate allows; a fragment-only navigation is not one. The scheme handler then serves from the bundle selected for it.
- A content rule list blocks every load outside the `eidola` scheme (and `data:`/`blob:`), so another origin is refused by the shell and not only by the CSP.
- Durable means `F_FULLFSYNC` on the file and on its directory (plain `fsync` stops at the drive's write cache on Apple platforms; `F_FULLFSYNC` works on a directory fd on APFS). Files are `completeUntilFirstUserAuthentication` and excluded from backup; the reasons are in the contract.
- Debug builds forward the page's console to the system log (subsystem `ai.eidola.shell`) and make the webview inspectable; Release builds do neither.
- **The iOS shell cannot be reproducibly verified.** Apple re-signs and encrypts what the App Store distributes, so nobody can compare an installed binary with a build of this source. Its smallness is the mitigation: what it may do is this document and the contract, and it is short enough to read.

## Android

- The build chain is `aapt2 link` (manifest only) → `kotlinc` against `android.jar` alone → R8 (our classes plus `kotlin-stdlib.jar`, shrunk, `keep.pro`) → `tools/ApkZip.java` (sorted, stored, one fixed timestamp) → `zipalign` → `apksigner` (v2 + v3, alignment preserved).
- **The App Bundle** (`--aab`, release only) is what Play takes for a new app. Nothing is compiled twice: `aapt2 link --proto-format` links the same manifest and resources as protocol buffers, `tools/aab_module.py` lays them out with the APK's own `classes.dex` and bundle as a bundletool module, `bundletool build-bundle` wraps it, and `tools/zip_stored.py` rewrites bundletool's deflated entries stored, in name order, with one timestamp, so the bytes do not depend on the JDK's zlib. Play rebuilds the APKs it delivers from it and signs them with its own key: what users verify is the APK we publish; a Play install compares by content (`tools/compare_apk_contents.py`: `classes.dex` and `assets/` must be our bytes; the manifest and resource table carry Play's split configuration, and the signature is Play's).
- `--debug` makes the APK debuggable (console in logcat, inspectable WebView) and names it `ai.eidola.mobile.dev`; otherwise the variants are identical.
- Edge to edge, the WebView reports no safe areas and does not resize for the keyboard, so `MainActivity` pads its container by the system bars, the cutout and the IME.
- The package is `ai.eidola.mobile` for a release build and `ai.eidola.mobile.dev` for `--debug` (aapt2 renames the manifest's package), so a dev build installs beside a release one. **The Play package name is permanent once published**: an app under another name is another app, with no upgrade path from this one.

### Reproducibility

The unsigned APK is a function of the source, the bundle and these pinned inputs, fetched once by `tools.sh` into `~/.cache/eidola/android-tools` and checked by SHA-256 before use (Google's repository manifest lists their SHA-1s, which match too). The local SDK is not used to build.

| Input | Version | SHA-256 of the archive |
|---|---|---|
| build-tools (macOS) | 36.1.0 | `cc56be1d4ea95041f32f4ebf3a4f5c6b990c7749d706852395006b5654eb82ff` |
| build-tools (Linux x86_64) | 36.1.0 | `a7b5889e4a79fcf3b0976bef40d401f4240fb1eed891d9d91169da1111e11d78` |
| platform `android-36` | r02 | `37607369a28c5b640b3a7998868d45898ebcb777565a0e85f9acf36f29631d2e` |
| `kotlin-compiler` | 2.3.10 | `c8d546f9ff433b529fb0ad43feceb39831040cae2ca8d17e7df46364368c9a9e` |
| `bundletool-all` (jar, App Bundle only) | 1.18.3 | `a099cfa1543f55593bc2ed16a70a7c67fe54b1747bb7301f37fdfd6d91028e29` (the digest GitHub records for the release asset) |

The JDK is not an input: kotlinc compiles against `android.jar` only (`-no-jdk`), and R8 and the zip writer produce the same bytes on any JDK from 17 up. The same source gives the same APK on macOS (arm64, JetBrains Runtime 25) and on Linux (x86_64, Temurin 17 and 21), so `reproducible.sha256` holds on both.

A **signed** APK differs on every signing (ECDSA signatures are randomized). `tools/strip_apk_signature.py SIGNED OUT` removes the v2/v3 signing block and the padding apksigner adds before it, restoring the unsigned bytes exactly; anyone can compare a published APK with their own build that way. This holds for an APK **we** publish. Play App Signing rebuilds what Play delivers (its own signature, a stamp, manifest additions), so a Play install compares by content, file by file, with Play's additions excluded.

**Signing a release** is a person's step at release time, never CI's: the release key signs the APK we publish (`apksigner`, v2 + v3, alignment preserved, as above), and Play's upload key signs the App Bundle. The steps, and how the signed APK's hash and certificate reach the human attestation, are in `releases/README.md` → Signing the Android shell.

## CI

`.github/workflows/mobile-shells.yml` runs on changes under `mobile/` and `crates/eidola-mobile/`, and to what pins the bundle's build (`Cargo.lock`, `rust-toolchain.toml`, `.cargo/config.toml`), on the `mobile` branch: it builds the Android shell twice on Linux with the pinned tools (APK and App Bundle), checks the strip round trip and the App Bundle's universal APK, and checks `reproducible.sha256`; and it builds the app bundle twice from scratch (`bundle-repro.sh`). `.github/workflows/mobile-bundle-release.yml` rebuilds the release bundle on a hosted runner and requires `bundle-release.digest`. On `main`, the artifact chain's `mobile` job builds the release bundle and the unsigned APK and App Bundle over it, for the manifest (from schema 6) and the release-time signing step. iOS is the local gate above. `rust-checks` lints `eidola-mobile` for wasm32 (with and without `self-test`) on every Rust change.
