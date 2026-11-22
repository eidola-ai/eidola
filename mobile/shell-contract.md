# The mobile shell contract

**Contract version 2.** This document is the whole interface between Eidola's native mobile shells (`mobile/ios`, `mobile/android`) and the app bundle they host (the wasm build of the Rust app, or any page, such as `mobile/contract-test/`). Both shells implement exactly this, and the app may rely on nothing else.

## Roles

The **shell** is the native app the stores distribute. It is part of the platform: small, dependency-free, and dumb. It serves a bundle's bytes, moves raw TCP bytes, keeps files in the app sandbox, stores bundles the app hands it, shows which bundle is running, fits the webview to the screen, and hands `https:` links to the system browser. It never fetches anything from the network on its own, never verifies or judges any bytes, and holds no product logic.

The **app** is the bundle running in the system webview. TLS, attestation, update verification, and every other trust decision happen in the app. A byte the app hands the shell has already been judged by the app; a byte the shell hands the app is whatever the peer or the disk held.

## Versioning

- The contract has one integer version, returned by `shell.hello`. This document describes version 2. Version 2 added `shell.openExternal`.
- Changes are additive only: a new method, a new optional parameter, a new member in a result, a new error code. A new version never removes a method, a member or a code, and never changes what an existing one means. The version increases with each addition.
- The app detects what a shell supports with `shell.hello`'s `contract` and `methods`, and must ignore result members it does not know.

## Origin and serving

The bundle is served on an origin the shell owns:

| Platform | Origin | Mechanism |
|---|---|---|
| iOS | `eidola://app` | `WKURLSchemeHandler` for the `eidola` scheme |
| Android | `https://app.eidola.invalid` | `WebViewClient.shouldInterceptRequest` (the `.invalid` name never resolves) |

Both are secure contexts. The app must read its origin from `location.origin` and never hard-code it.

- **Paths.** A request is served only if its path is empty (the bare origin, as `location.origin` spells it) or `/`, both of which mean `/index.html`, or a **bundle path**: one or more segments joined by `/`, each matching `[A-Za-z0-9._-]+` and not starting with `.`. Query strings and fragments are ignored. Any other path, and any path the selected bundle does not hold, is a `404`. Only `GET` is served; other methods get `405`.
- **No mixing.** Every response of one document load comes from one bundle, the one selected when that document's main-frame navigation began (see [Bundle store](#bundle-store)). A file missing from it is a `404`, never a file from another bundle.
- **Headers.** Every response carries:
  - `Content-Type` by extension: `html` `text/html; charset=utf-8`, `js` and `mjs` `text/javascript; charset=utf-8`, `wasm` `application/wasm`, `css` `text/css; charset=utf-8`, `json` and `map` `application/json`, `txt` `text/plain; charset=utf-8`, `svg` `image/svg+xml`, `png` `image/png`, `jpg` and `jpeg` `image/jpeg`, `webp` `image/webp`, `ico` `image/x-icon`, `woff2` `font/woff2`, `woff` `font/woff`, `ttf` `font/ttf`, `otf` `font/otf`; anything else `application/octet-stream`.
  - `Content-Security-Policy: default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'`. Inline scripts are refused, so a bundle loads its scripts from files. Inline styles are allowed because renderers set them and they cannot run code.
  - `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`.
- **Other origins are refused** by the shell, not only by the CSP: Android answers every request to another origin with `403`, and iOS installs a content rule list that blocks every load outside the `eidola` scheme. A main-frame navigation away from the origin is cancelled. Nothing the page does makes the shell fetch anything, and the webview makes no requests of its own (Safe Browsing and fraudulent-site lookups are off, and Android's WebView metrics are opted out).
- **Document loads.** The shell selects a bundle at every main-frame navigation to its origin: the first load at launch, `location.reload()`, and the shell's own reload after the web content process dies. A new document load also closes every socket the previous document opened.

## The bridge

The app calls the shell with **requests** and gets exactly one **reply** per request. There are no unsolicited messages: data from a socket arrives as the reply to a `tcp.receive`.

```json
{"id": 7, "method": "files.read", "params": {"name": "config.toml"}}
{"id": 7, "ok": true, "result": {"data": "aGVsbG8="}}
{"id": 7, "ok": false, "error": {"code": "not_found", "message": "no such file"}}
```

- `id` is a number the app chooses; the reply echoes it. `params` may be omitted when a method takes none.
- Every message is one JSON text, sent as a string.
- **Bytes** travel as standard base64 with padding (RFC 4648 section 4).
- **Integers** (offsets, sizes, ports, socket ids) are JSON numbers holding non-negative integers no larger than 2^53 - 1.
- **Order.** All `files.*` and `bundle.*` requests are applied one at a time, in the order they arrived, on one queue: a request's effects are complete before its reply is sent and before the next one starts. `tcp.*` requests on one socket are applied in arrival order; sockets are independent of each other and of the file queue.

### Transport on iOS

The shell registers a `WKScriptMessageHandlerWithReply` named `eidola` in the page content world.

```js
const reply = await window.webkit.messageHandlers.eidola.postMessage(JSON.stringify(request));
// reply is the reply's JSON text
```

A message from any frame other than the main frame of the shell's origin is refused (the promise rejects).

### Transport on Android

The shell exposes one JavaScript interface, `eidolaShellAndroid`, with one method, `open()`. Calling it makes the shell create a `WebMessageChannel` and post the string `eidola-shell` to the main frame, restricted to the shell's origin, carrying one port. Requests go out as strings on that port; replies come back as strings on it.

```js
const port = await new Promise((resolve) => {
  addEventListener("message", function once(e) {
    if (e.data === "eidola-shell" && e.ports.length === 1) { removeEventListener("message", once); resolve(e.ports[0]); }
  });
  eidolaShellAndroid.open();
});
port.onmessage = (e) => handleReply(JSON.parse(e.data));
port.postMessage(JSON.stringify(request));
```

Calling `open()` again replaces the channel; replies to requests sent on the old one are dropped.

### Reference bridge

`mobile/contract-test/shell.js` implements both transports behind one call, `shell.call(method, params)`, which resolves to `result` or rejects with an `Error` carrying `code`. The app's own host bridge implements the same.

## Errors

A failed request replies `{"ok": false, "error": {"code", "message"}}`. `message` is for logs only; the app acts on `code`:

| Code | Meaning |
|---|---|
| `bad_request` | The message is not valid JSON, or a parameter is missing, of the wrong type, out of range, or not a valid name or path. |
| `unknown_method` | The shell does not implement this method (see [Versioning](#versioning)). |
| `not_found` | The named file, socket or bundle version does not exist. |
| `invalid_state` | The request is not allowed now (each method lists when). |
| `connect_failed` | `tcp.connect` could not open the connection: refused, unreachable, the name did not resolve, or it timed out. |
| `closed` | The socket was closed while the request was pending. |
| `io` | The operating system failed the operation (disk full, a reset connection, …). |

## Methods

### `shell.hello`

The version handshake. Params: none. Result:

- `contract`: the contract version (2).
- `platform`: `"ios"` or `"android"`.
- `shell`: the shell's own version string, for display and logs.
- `methods`: every method this shell implements, sorted.

### `shell.openExternal`

Params: `url`. Result: `{}`, once the operating system has accepted the URL.

Hands the URL to the system's own browser (or the app the system routes it to): `UIApplication.open` on iOS, an `ACTION_VIEW` intent on Android. It is never loaded in the shell's webview and never in a browser inside the app (no `SFSafariViewController`, no Custom Tabs), so the page the user sees there is the browser's, with its own address bar, and nothing of it reaches the app.

It exists because some things the app offers happen on the web, purchases first: the app links out to them. The app decides which URLs to open; the shell only refuses what is not an absolute `https:` URL: the scheme `https` (any case), a host, and nothing but printable ASCII (`bad_request` otherwise; percent-encode anything else). The shell neither fetches the page nor looks at it. `io` when the system opened nothing.

### TCP pipe

A raw TCP byte stream the app drives. The shell never does TLS and never looks at the bytes. Socket ids are chosen by the app and must be unique among its open sockets; that lets the app close a socket whose connect is still pending.

| Method | Params | Result | Notes |
|---|---|---|---|
| `tcp.connect` | `socket`, `host`, `port` | `{}` | `host` is a DNS name or an IP literal (no brackets); `port` 1 to 65535. Resolves once connected. The shell gives up after 30 seconds (`connect_failed`). `invalid_state` if the id is in use. Closing the socket while pending fails it with `closed`. |
| `tcp.send` | `socket`, `data` | `{}` | Resolves once every byte was handed to the operating system. Sends on one socket go out in request order. `invalid_state` before connect completes or after `tcp.shutdown`. |
| `tcp.receive` | `socket`, `max` (optional, default 65536, at most 1048576) | `{"data": "…"}` or `{"eof": true}` | Waits for at least one byte and returns at most `max`. `eof` once the peer closed its side; every later receive returns `eof` again. One receive at a time per socket (`invalid_state` otherwise). The shell reads from the network only while a receive is pending, so the app paces the stream. |
| `tcp.shutdown` | `socket` | `{}` | Closes the sending side (TCP FIN); receiving continues. |
| `tcp.close` | `socket` | `{}` | Closes both sides and forgets the id. Pending requests on it fail with `closed`. An unknown id is not an error. |

Errors on a connected socket (a reset, a timeout) fail the pending request with `io`; the app closes the socket.

### Files

A flat directory of files inside the app sandbox, owned by the app: its database, its write-ahead log, its config. Nothing else in the shell reads or writes it, and it is not reachable from the origin.

**Names** are 1 to 255 characters of `[A-Za-z0-9._-]`, not starting with `.`. There are no directories.

| Method | Params | Result | Notes |
|---|---|---|---|
| `files.read` | `name` | `{"data": "…"}`, or `{"data": null}` when missing | The whole file. |
| `files.write` | `name`, `data`, exactly one of `offset` or `append: true`, `sync` (optional) | `{"size": n}`, the file's length after the write | Creates the file when missing. A write past the end extends the file; the gap reads as zeros. |
| `files.truncate` | `name`, `size`, `sync` (optional) | `{}` | Sets the length, creating the file when missing. Growing it adds zeros. |
| `files.sync` | `name` | `{}` | Makes the file's contents, length and existence durable. A missing file is not an error. |
| `files.rename` | `from`, `to` | `{}` | Atomically replaces `to` with `from`. `not_found` when `from` is missing. Always durable, contents included (see below). |
| `files.remove` | `name` | `{}` | A missing file is not an error. Always durable. |
| `files.list` | none | `{"names": [...]}` | Every file, sorted. |

**Durability.** Every request is applied in full before its reply, so its effect survives the app being killed. Surviving the device losing power is a stronger promise, made only by the requests below, and only once they reply. To **flush** a file is to force its contents and length to stable storage; to flush a directory is to force its names there. iOS flushes both with `F_FULLFSYNC`, because plain `fsync` there stops at the drive's write cache; Android uses `fsync`.

| Request | Durable once it replies |
|---|---|
| `files.write`, `files.truncate` with `sync: true` | The file's contents and length, and its name. The shell flushes the file, then the directory, whether or not this request created the name. |
| `files.write`, `files.truncate` without `sync` | Nothing yet. A later `files.sync` (or a synced write) of the same file makes it durable. |
| `files.sync` | The file's contents, length and name (the file, then the directory). A missing file: the directory only. |
| `files.rename` | The file now named `to` holds `from`'s contents, and both names' changes. The shell flushes `from` before renaming it and the directory after, so an unsynced write followed by a rename is the whole atomic-replace idiom: no separate sync is needed. |
| `files.remove` | The name's removal (the directory). |
| `bundle.seal` | Every file and directory of the version, and the version's own name, before the state records it as sealed. |
| `bundle.activate`, `bundle.confirm`, `bundle.remove` | The store's state (written to a temporary file, flushed, renamed over the old one, directory flushed). |
| `bundle.begin`, `bundle.write` | Nothing: a version is durable only once sealed. |

After a power loss, a file whose last change was not durable holds some earlier or partial state of it; the app's storage layer orders its writes and flushes so every such state reopens consistently. `data` in one request is at most 8 MiB decoded.

### Bundle store

The shell keeps bundle versions the app hands it and a pointer to the one to boot. It judges nothing about their bytes: the app verified them before staging and decides what to activate.

A **version** name is 1 to 64 characters of `[A-Za-z0-9._-]`, not starting with `.`. The name `embedded` means the bundle built into the shell; it is always present, never staged and never removed. A staged bundle's files are named by bundle paths, as above.

| Method | Params | Result | Notes |
|---|---|---|---|
| `bundle.begin` | `version` | `{}` | Starts staging `version`, empty. Restarts it when it is already staging. `invalid_state` when it is sealed (remove it first) or `embedded`. |
| `bundle.write` | `version`, `path`, `offset`, `data` | `{}` | Writes into a staging version's file at `offset`, creating it. `invalid_state` unless `version` is staging. At most 8 MiB per request. |
| `bundle.seal` | `version` | `{"digest": "…", "files": n, "bytes": n}` | Makes the version durable (see Durability above), computes the bundle's [digest](#bundle-digest) from the bytes on disk, and marks the version sealed and read-only. `invalid_state` unless staging. |
| `bundle.activate` | `version` | the state, as `bundle.state` | Makes a sealed version (or `embedded`) the one the next document load boots. Durable before the reply. Activation is **provisional**; see below. |
| `bundle.confirm` | none | the state | Tells the shell the bundle now running booted. Confirms it when it is the active bundle on its trial; otherwise does nothing. |
| `bundle.remove` | `version` | `{}` | Deletes a staging or sealed version. `invalid_state` for `embedded`, the active version, the previous confirmed version, and the version now being served. `not_found` when unknown. |
| `bundle.state` | none | see below | |

`bundle.state` returns:

- `serving`: `{"version", "digest", "confirmed"}` of the bundle this document was served from.
- `active`: the version the next document load boots (`"embedded"` when no staged version is active).
- `confirmed`: whether the active version has been confirmed.
- `previous`: the last confirmed version the shell falls back to when the active one fails its trial (`"embedded"` when none).
- `failed`: the last version that failed its trial, or `null`.
- `versions`: every stored version, sorted by name, as `{"version", "sealed", "digest"}` (`digest` is `null` while staging).

#### Provisional activation

A bad update must not be able to brick the app, so an activated version boots once on trial and stays active only if the app confirms it booted.

The shell keeps, durably: `active`, `confirmed`, `tried`, `previous` and `failed`. At first launch `active` is `embedded`, `confirmed` is true, `previous` is `embedded`.

- **`bundle.activate(v)`**, `v` sealed: if the current active version is confirmed, it becomes `previous`. Then `active = v`, `confirmed = false`, `tried = false`. Activating `embedded` sets `active = previous = embedded`, `confirmed = true`.
- **Each document load:**
  1. If `active` is unconfirmed and not yet `tried`: set `tried = true` (durably, before serving) and boot `active`. This is its trial.
  2. If `active` is unconfirmed and already `tried`, its trial failed (the app was killed, crashed, hung until relaunch, or reloaded without confirming): `failed = active`, `active = previous`, `confirmed = true`, `previous = embedded`, durably; boot the new `active`.
  3. If the active version's files are gone or it has no `index.html`: `failed = active`, `active = previous = embedded`, `confirmed = true`, durably; boot `embedded`.
- **`bundle.confirm`** during the trial: `confirmed = true`, `previous = embedded`. The older version's files stay until the app removes them.

The app's part: after a verified update is staged, sealed and approved, call `bundle.activate`, then `location.reload()`; on every boot, call `bundle.confirm` as soon as the app is up (it does nothing outside a trial). A failed version stays stored, and the app can see it in `failed`.

## Bundle digest

The shell computes a digest of the bundle it serves, from the bytes it serves, and shows it in its own native UI. It is a fingerprint, not a trust decision: the shell compares it with nothing.

1. List every file of the bundle by its bundle path.
2. Sort the paths by their bytes (they are ASCII).
3. For each, write the line `<hex SHA-256 of the file's bytes><two spaces><path><LF>`. This is the format of `sha256sum`.
4. The digest is the lowercase hex SHA-256 of all those lines concatenated.

It can be reproduced from a bundle directory with `mobile/tools/bundle_digest.py DIR`, or with coreutils:

```sh
cd DIR && find . -type f | sed 's|^\./||' | LC_ALL=C sort | xargs sha256sum | sha256sum
```

## Native UI

The shell's one piece of UI is the **bundle sheet**, drawn natively and so out of the bundle's reach. It shows the serving bundle's version, its digest (in four groups of sixteen hex digits), whether it is confirmed or on trial, the active version when it differs, and the shell and contract versions, with a button that copies the digest.

It opens from the app icon: a long press on the home-screen icon offers **Bundle digest** (a home-screen quick action on iOS, a launcher shortcut on Android), which opens the app and shows the sheet over it.

## Screen and keyboard

- **Android** draws edge to edge, and its WebView reports zero safe-area insets and does not shrink for the soft keyboard. The shell therefore pads the WebView by the system bars, the display cutout and the keyboard: the page's viewport is exactly the visible, unobstructed area, and `env(safe-area-inset-*)` is zero.
- **iOS** extends the viewport under the status bar and home indicator when the page's viewport meta has `viewport-fit=cover`; the page reads the insets from `env(safe-area-inset-*)` and follows the keyboard with `visualViewport`. The shell adds nothing. The keyboard's input accessory bar (previous, next, done) stays: hiding it needs a private WebKit class.

## Storage and data protection

- The file directory and the bundle store live in the app's private storage: `Library/Application Support/Eidola/` on iOS, `noBackupFilesDir` on Android.
- Neither is backed up (`isExcludedFromBackup` on iOS; `noBackupFilesDir` and `allowBackup="false"` on Android). A restored copy of the wallet on another device would hold credentials already spent.
- **iOS data protection class: `completeUntilFirstUserAuthentication`.** Files are encrypted at rest until the user unlocks the device once after it boots, and readable from then on. The stricter `complete` class would make every write fail the moment the device locks, and a file stream that fails stops for good (the app's storage contract), so a turn still streaming or a background task would lose its durability. Before the first unlock nothing of the app's runs.
- Nothing durable lives in webview storage (local storage, IndexedDB, OPFS): the webview evicts it. iOS gives the webview a non-persistent data store and Android disables DOM storage, so such storage lasts at most until the app exits.

## What the shell never does

- Fetch anything from the network on its own, load another origin in its webview, or serve one. (`shell.openExternal` hands a URL to the system browser; the shell itself loads nothing.)
- Verify, sign, compare or interpret any bytes: of a bundle, a file, or a socket.
- Choose a bundle by anything but the state above.
- Keep any state of the app's besides the files the app wrote.
