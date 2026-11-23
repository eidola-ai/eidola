# The mobile app

The mobile app is the Eidola client built for phones: the same core the desktop app and CLI run, compiled to WebAssembly and run in the phone's system webview, inside a small native shell. This page says what that arrangement lets a user verify on each platform, and what it does not. How the desktop client decides what to trust is in [the client](client.md); how a release is hashed is in [verification](verification.md).

> [!NOTE]
> The mobile app is in development and is not distributed through either store. Its interface covers conversations: a list of them, and a conversation with posts rendered as formatted text, answers streamed with their reasoning, replies that branch, and a composer that works with the system keyboard and its input methods, in English, Spanish, French and both Chinese scripts. It has no screens yet for an account, the wallet, buying credits, settings or updates. No release records a mobile artifact yet. Where this page describes a mechanism the app does not use yet, it says so; such mechanisms are exercised by the core's tests and by a self-test that runs a bundle under each shell in the iOS Simulator and the Android emulator.

## The shape

The app is two pieces with very different jobs.

- **The bundle** is the app: the shared core compiled to WebAssembly, and the page that loads it. Every decision the app makes is made here: TLS, enclave attestation, release verification, the wallet, the local database.
- **The shell** is the native app a store would distribute, written for each platform with the system's own frameworks and no third-party code. It serves the bundle to the webview from an origin of its own, carries raw TCP bytes, keeps the app's files, stores bundles the app hands it and chooses which to boot, shows which bundle is running, fits the webview to the screen, and opens web links in the system browser. It fetches nothing from the network on its own, judges no bytes, and holds no product logic. Everything it does is specified in [the shell contract](../mobile/shell-contract.md), short enough to read in full, and a test page checks both shells against it.

The split exists because neither store distributes the bytes a developer builds. Apple re-signs and encrypts what the App Store delivers, and Google builds and signs what Google Play delivers, so a store binary cannot be held to a hash of a build from source the way a desktop download is. Store review would also gate every release. So the part that must be verifiable is not the store's binary but the bundle, which can reach the phone as data and be checked by the app itself. Each shell build embeds a complete bundle, so the app runs from first launch without downloading any code.

## What can be verified

| Piece | iOS | Android |
| --- | --- | --- |
| The bundle | Built reproducibly; its digest is shown by the shell | The same |
| The shell | Cannot be verified | The unsigned APK reproduces byte for byte, and a signed one reduces to it; a Play install could be compared only file by file |
| The operating system and its webview | Trusted, as the desktop app trusts its operating system | The same |

### The bundle

`mobile/build-bundle.sh` builds the bundle from source and prints its **digest**: a single SHA-256 over the sorted list of its files and their SHA-256s, defined in [the shell contract](../mobile/shell-contract.md#bundle-digest). Two clean builds produce identical files (`mobile/bundle-repro.sh`, which CI runs). The release manifest's format has a row for the bundle (its files with their lengths and SHA-256s, the SHA-256 of the archive they are published in, and the digest), which clients accept; no release emits it yet.

The shell computes the same digest from the files it serves. To see it, long-press the app's icon on the home screen and choose **Bundle digest**. The shell shows its **bundle sheet**, drawn natively over the app: the running bundle's version, its digest, whether it has been confirmed or is booting on trial, the version the next launch will boot when that differs, and the shell's own version and contract version, with a button that copies the digest. The sheet is opened from the home screen, outside the app, so the bundle cannot draw it or put anything in front of it.

The digest of any bundle directory can be recomputed with nothing but coreutils:

```sh
cd mobile/build/bundle && find . -type f | sed 's|^\./||' | LC_ALL=C sort | xargs sha256sum | sha256sum
```

`shasum -a 256` in place of both `sha256sum`s gives the same result, and `mobile/tools/bundle_digest.py DIR` computes it too.

The sheet is the shell's statement of what it serves, so it is exactly as trustworthy as the shell that shows it.

### The Android shell

The Android shell's unsigned APK is a function of its source, the bundle it embeds and build tools pinned by hash: the same source gives the same APK on macOS and Linux, which CI checks. A signed APK differs only by its signature, and removing the signature restores the unsigned bytes exactly, so a signed APK, or one taken off a phone, can be held to a build from source:

```sh
adb shell pm path ai.eidola.mobile             # where the installed APK is
adb pull <path> eidola.apk
python3 mobile/android/tools/strip_apk_signature.py eidola.apk unsigned.apk
sha256sum unsigned.apk                         # compare with a build from the same source
```

`mobile/android/build.sh --bundle DIR` builds the unsigned APK around a bundle directory, and `mobile/android/repro.sh` checks the whole round trip.

That byte-for-byte check holds for an APK signed by its publisher and nothing else. What Google Play delivers is built and signed by Google, which adds its own signature and its own entries, so an install from Play could be compared with a build only file by file, leaving out what Google added. That is a weaker check: what is left out is not checked.

### The iOS shell

**The iOS shell cannot be verified.** Apple re-signs and encrypts the binary the App Store delivers, so nobody, Eidola included, can compare an installed copy with a build of its source.

What bounds its role is its design: it carries only ciphertext, because TLS ends inside the bundle; it fetches nothing on its own; it makes no trust decision; and it is small and has no third-party code. Those are properties of its source. A user who runs the App Store's copy is trusting that the copy is that source. The webview that runs the bundle, WebKit on iOS and the system WebView on Android, was always in the trusted base, as the operating system is for the desktop app.

## Attestation

TLS and enclave attestation run inside the bundle. The shell's part is a raw TCP pipe: the bundle asks it to open a connection, and from then on hands it bytes that are already encrypted and receives bytes it decrypts itself. The shell never terminates TLS and never sees a request or an answer in the clear.

The check is the desktop client's own ([per-handshake attestation](client.md#per-handshake-attestation-no-caching)): the same verifier code, against the same trust root compiled into the bundle, with the same refusal when it fails. Over the pipe nothing is pooled: every request opens its own connection, and a request to Eidola's gateway is attested on it. Connections that carry no attestation, such as a release download, are checked against the WebPKI roots compiled into the bundle. The app as it stands makes no request at boot; the self-test makes attested requests over each shell's pipe and is refused by an enclave whose measurement is not pinned.

The webview gives the bundle no other way out. The shell serves only its own origin and refuses every other one, and the page's content security policy allows connections to that origin alone, so the app cannot fall back to the browser's own networking around the pipe.

## Updates

The app does not offer updates yet: nothing in it checks for one. The core implements the mobile update, and the self-test runs it under both shells.

The core finds and verifies the next release exactly as the desktop client does ([releases](releases.md#how-the-client-verifies-a-release)): CI's signature on the manifest, the human attestations from pinned keys, their entries in the Rekor transparency log, every claim re-rendered from the pinned templates, and a comparison of the release's claims with what this client expects. A manifest in a schema it does not accept is refused. It records no installed release, so it would be offered the oldest release newer than its own version, without the continuity check that anchors a release to the one installed ([first-install downgrade](gaps.md#first-install-downgrade)).

Verifying a release downloads only the release's documents. The bundle is fetched only by a separate approval step, the only path by which a downloaded bundle reaches the shell. That step downloads the bundle's archive, requires its hash to be the manifest's, unpacks it, and requires every file to be one the manifest lists, of the listed length and hash, and every listed file to be there. Only after all of that does it hand the files to the shell. The shell stores them and computes the digest of what it stored, which the core requires to be the manifest's; on any difference the stored copy is removed.

Activation is provisional, in the shell. It boots a newly activated bundle once, on trial, and keeps it only if that bundle confirms it booted, which the app does at every boot once it has opened the profile, made it durable and drawn its first screen. A bundle that never confirms, because it crashed, hung or was killed, is replaced at the next launch by the last confirmed bundle, or the embedded one, and the failure is recorded; the core does not stage that version again. The core can also return the shell to the embedded bundle. The shell judges nothing about the bytes in any of this: whether a bundle boots is the only thing it observes.

The self-test verifies a real signed release over the shell's pipe. Because no signed release records a bundle yet, it then stages a bundle described by test evidence: downloaded, checked, stored, activated, confirmed after a relaunch with the shell's digest matching; a broken bundle falling back to the confirmed one; and the return to the embedded bundle.

## Storage

The app's profile (its database, which holds conversations and the wallet, and its settings, which hold the account credentials) lives in files in the app's private storage, written through the shell's file API. Nothing durable lives in webview storage, which the platform may evict: the shell gives the webview no persistent storage at all. The files are excluded from device backups, because a copy of the wallet restored onto another device would hold credentials that have already been spent. On iOS they are encrypted at rest until the device is first unlocked after it starts.

The core works on the database in memory and streams every write to the shell, in order. **No request leaves the device until every write made before it is durable**, synced to storage so that it survives a crash or a power loss. If the shell does not make them durable in bounded time, the request is refused and never sent. This is what keeps credits honest across a crash: the record of an issuance is on disk before the request for it is sent, and the record of a spend before its proof, so after any crash the app finds every credential the server may have seen.

The app does not encrypt these files itself: they are as private as the platform keeps the app's storage ([privacy guarantees §8.1](privacy-guarantees.md#8-bounded-claims-what-this-document-does-not-promise)).

## What the shell is trusted with

The shell as written carries ciphertext, keeps files the app writes, serves the bundle it stored and shows its digest. It cannot read a conversation off the network, because it never holds a TLS key, and it cannot attest an enclave that is not one, because attestation is the hardware's signature over the enclave's identity and the app checks it itself.

A shell that is not what its source says is not bounded by its design. It holds the profile's files, so it could read conversations, the account credentials and the wallet. It controls the webview, so it could add code to the page. It chooses which bundle to serve and draws the sheet, so it could serve other code, one that skips attestation, and show the digest of the real bundle. That is the reach the operating system and its webview already have over every app; a modified shell gives the same reach to whoever built it. It reaches only the phone it runs on: it cannot make an enclave attest to anything else, read anyone else's traffic, or produce a release the desktop clients would accept.

On Android, the reproducible APK lets a user establish that the shell they run is built from the published source. On iOS nothing can, so there the shell is trusted the way the platform is, and that trust extends to Eidola, which would build and upload it ([threat model](threat-model.md#n1-a-compromised-local-environment)).

## Store distribution

Each shell build embeds a complete bundle, so what a store reviews is the working app, and any later bundle runs in the system webview the same way. Apple's developer agreement allows an app to download code that runs in an interpreter when it does not change the app's primary purpose or bypass the system's security, and Google Play's rules on downloaded code exempt code that runs in a webview.

The app offers no purchase, and Eidola sells nothing through either store's payment system: credits are bought on Eidola's website. Both stores let an app on their United States storefronts link to a purchase on the web. The shells give the app one way to open anything outside its own origin, `shell.openExternal`, which the app does not use yet: it accepts only `https:` links and hands them to the operating system's browser, never to its own webview or a browser inside the app, so a page opened that way is the browser's, with its own address bar, and nothing on it reaches the app.
