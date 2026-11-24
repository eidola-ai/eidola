# Device checklist

What to prepare before a shell build goes onto a physical device, and the exact steps of a session. Everything that can be proven in the Simulator or the emulator is proven there first; a device session proves only what they cannot.

## What only a device proves

- WebGPU on real hardware (the Simulator and the emulator fall back to WebGL2).
- The real keyboard and IME over the webview, and the safe areas of a real screen.
- The pipe over the device's own network, including the iOS Local Network permission prompt.
- The app surviving a lock and unlock mid-run (files are `completeUntilFirstUserAuthentication`).
- `F_FULLFSYNC` cost on real storage.

## Before the session (on the Mac)

- [ ] `just mobile-ios-test` and `just mobile-android-test` pass on this commit.
- [ ] Xcode is signed in to an Apple ID on team `JPK236247S` (Xcode → Settings → Accounts); the build uses automatic signing with the bundle id `ai.eidola.mobile.dev` (`IOS_TEAM` names another team).
- [ ] Builds made ahead, one per bundle, each with the digest `build.sh` prints noted:
  - the contract test page: `mobile/ios/build.sh --device --bundle mobile/contract-test`
  - the app bundle: `mobile/ios/build.sh --device` (it builds `mobile/build/bundle` first)
- [ ] The Mac's LAN address noted (`ipconfig getifaddr en0`), and the echo server ready to start on it: `python3 mobile/tools/echo_server.py 7777 0.0.0.0`.
- [ ] A file `contract-echo.txt` holding `<LAN address>:7777`, for the test page to find the echo server.

## On the device

- [ ] Developer Mode on (Settings → Privacy & Security), paired and trusted; it shows in `xcrun devicectl list devices` (note its identifier as `DEVICE`).
- [ ] On the same Wi-Fi as the Mac; Auto-Lock set long for the session.
- [ ] No other build of `ai.eidola.mobile.dev` installed, or one that may be replaced.

## The session

1. Install and launch once. This first run has no echo server and fails its pipe checks; it only creates the shell's directories.

   ```sh
   xcrun devicectl device install app --device "$DEVICE" mobile/ios/build/DerivedData/Build/Products/Debug-iphoneos/EidolaShell.app
   xcrun devicectl device process launch --device "$DEVICE" ai.eidola.mobile.dev
   ```

2. Start the echo server, give the test page its address, and restart the run (an empty `contract-progress.json` starts a fresh one):

   ```sh
   python3 mobile/tools/echo_server.py 7777 0.0.0.0 &
   : >| contract-progress.json
   for f in contract-echo.txt contract-progress.json; do
     xcrun devicectl device copy to --device "$DEVICE" --domain-type appDataContainer \
       --domain-identifier ai.eidola.mobile.dev --source "$f" \
       --destination "Library/Application Support/Eidola/files/$f"
   done
   xcrun devicectl device process launch --terminate-existing --device "$DEVICE" ai.eidola.mobile.dev
   ```

3. Allow the Local Network prompt when it appears. That run may fail its pipe checks while the prompt was up; if so, repeat step 2's copy of `contract-progress.json` and the relaunch. Wait for `contract: PASS n/n` on screen (about a minute).
4. Copy the report back: `xcrun devicectl device copy from --device "$DEVICE" --domain-type appDataContainer --domain-identifier ai.eidola.mobile.dev --source "Library/Application Support/Eidola/files/contract-report.json" --destination .`
5. Long-press the app icon → **Bundle digest**. The sheet's digest equals the page's `serving embedded, digest …` line and the value `build.sh` printed.
6. Swipe the app away and reopen it: `restart: PASS`.
7. Lock the device during a fresh run (step 2) and unlock it after a minute: the run resumes and ends in `contract: PASS`. Writes from background work while locked are for the app to prove once it does any; the protection class allows them.
8. Install the real-bundle build and check what only the device shows: rendering backend (WebGPU), typing with the software keyboard, safe areas.

## Afterwards

- [ ] `xcrun devicectl device uninstall app --device "$DEVICE" ai.eidola.mobile.dev`, unless the session's owner says to keep it.
- [ ] Stop the echo server.

## Android devices

Not used yet: the emulator covers the contract. A device run is the same as `android/test-emulator.sh` with `adb -s <serial>` in place of the emulator, the echo address in `no_backup/files/contract-echo.txt` (`adb shell run-as ai.eidola.mobile.dev` writes it on a debuggable build), and only on a device its owner has offered.
