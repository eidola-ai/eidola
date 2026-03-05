# Crux Cross-Platform Architecture

## Context

The application needs to run on macOS today with iOS, Android, and potentially
web targets planned. The core logic — conversation state management, model
orchestration, event handling — should be written once and shared across
platforms. Only the UI rendering and platform-specific I/O should be native.

### Alternatives considered

- **Tauri / Electron:** Web-based UI rendered in a webview. Fast to prototype
  but produces non-native UIs. Tauri's Rust backend is appealing, but the
  webview layer adds complexity and doesn't feel native on Apple platforms.

- **Dioxus / Leptos:** Rust-native UI frameworks. Promising but immature for
  production Apple apps. Platform integration (accessibility, system menus,
  native controls) is incomplete.

- **SwiftUI only:** Native and polished, but the logic is locked to Apple
  platforms. Adding Android or web would require rewriting the core in another
  language.

- **Kotlin Multiplatform:** Strong cross-platform story but requires the JVM
  ecosystem. Doesn't integrate well with Rust.

## Decision

Use **Crux** (redbadger.github.io/crux) for cross-platform state management.
Crux implements an Elm-like architecture where the core is pure Rust and
platform shells handle all I/O and rendering.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Platform Shell (SwiftUI, Jetpack Compose, web, etc.)       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Sends Events to core via processEvent()              │  │
│  │  Handles Effects (Render, capabilities)               │  │
│  │  Updates UI from ViewModel                            │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │ FFI (UniFFI + bincode)
┌─────────────────────▼───────────────────────────────────────┐
│  Crux Core (crates/eidolons-shared)                          │
│  - Event: user actions + capability responses               │
│  - Model: private app state (not exposed to shell)          │
│  - ViewModel: public view state (serialized to shell)       │
│  - Effect: side-effects for shell to fulfill                │
│  - Capabilities: typed interfaces for side-effects          │
└─────────────────────────────────────────────────────────────┘
```

**Key invariant:** The core never performs side-effects directly. It emits
typed `Effect` values that the shell handles, then the shell sends results
back via `handleResponse()`. This makes the core deterministic and testable
without mocking platform APIs.

### FFI bridge: UniFFI

Mozilla's UniFFI generates Swift (and Kotlin, Python) bindings from Rust
`#[uniffi::export]` annotations. The bridge exposes three functions:

- `processEvent(data: [UInt8])` — send a bincode-serialized Event to the core
- `handleResponse(uuid: [UInt8], data: [UInt8])` — return a capability result
- `view() -> [UInt8]` — get the current bincode-serialized ViewModel

### Serialization: bincode

Events, Effects, and ViewModels cross the FFI boundary as bincode-serialized
byte arrays. Bincode is fast and compact. It is not human-readable, which
makes debugging the bridge harder, but the typed Crux codegen ensures
structural correctness at compile time.

### Two codegen pipelines

The FFI bridge requires two separate code generation steps:

1. **`uniffi-bindgen-swift`** generates the FFI bridge itself — Swift wrappers
   for `processEvent`, `handleResponse`, `view`, plus any service types
   exposed directly (e.g., `PerceptionService`).

2. **`crux_core::typegen`** generates Swift types for the domain model —
   `Event`, `Effect`, `ViewModel`, and their nested types — with bincode
   serialization conformance.

Both outputs are committed to the repository and CI-verified for freshness
(see [Reproducible Builds](reproducible-builds.md)).

### Capability implementations

Pure Rust crates in `crates/` implement capability logic (e.g.,
`eidolons-perception`). These are compiled into
`eidolons-shared` and re-exported via UniFFI as services that the Swift shell
can call directly.

### XCFramework distribution

The compiled `eidolons-shared` crate is packaged as an XCFramework (universal
static library for macOS arm64 + x86_64) for consumption by SwiftPM. The
macOS app's `Package.swift` references it as a binary target.

### Dedicated threads for non-Send types

WGPU backend types are not `Send+Sync`, which conflicts with UniFFI's
requirements. The solution: inference runs on a dedicated OS thread that owns
the model exclusively. The shell communicates with this thread via
`mpsc::channel`, and the thread has its own Tokio runtime for async operations
(model downloading). This pattern applies to any capability whose
implementation uses non-Send types.

## Consequences

**Benefits:**

- Core logic is written once in Rust, tested without platform dependencies.
- Platform shells are thin — they handle rendering and I/O, nothing else.
- The Elm-like architecture (Event -> update -> Effect) is predictable and
  easy to reason about.
- Adding a new platform means writing a new shell, not rewriting logic.
- Effects as data means side-effects are testable — the core's output is
  inspectable without executing I/O.

**Trade-offs we accept:**

- Two codegen pipelines add build complexity. Changing Rust types requires
  regenerating both UniFFI bindings and Crux types, then rebuilding the
  XCFramework.
- Bincode serialization is not human-readable. Debugging requires
  understanding the binary format or adding logging at the FFI boundary.
- Crux is a young framework with a small community. If it becomes unmaintained,
  the Elm-like pattern is simple enough to reimplement, but migration would
  be non-trivial.
- The dedicated-thread pattern for GPU inference adds complexity and means
  the core cannot directly await inference results — everything is
  asynchronous via Effects.
- The `eidolons-shared` crate compiles as `cdylib`, `staticlib`, and `lib`
  simultaneously, which increases build times.

## Future Considerations

- **iOS shell.** Crux supports iOS via the same UniFFI bridge. The
  XCFramework already targets the correct architectures. A SwiftUI iOS shell
  would share the same `eidolons-shared` core.
- **Android shell.** UniFFI generates Kotlin bindings. The core would be
  compiled as a shared library (`.so`) for Android's NDK targets.
- **Web shell.** Crux supports WebAssembly. The pure-Rust dependency stack
  (see [Pure Rust, Zero C Dependencies](pure-rust-zero-c-dependencies.md)) makes WASM
  compilation feasible.
