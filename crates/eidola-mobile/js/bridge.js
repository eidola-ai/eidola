// The webview's half of the bridge to the shell (mobile/shell-contract.md → The
// bridge): it moves request and reply texts and nothing else. Ids, JSON, base64
// and every error are the Rust side's (src/web.rs).

let port = null;
let queued = [];

/** "ios", "android", or undefined outside a shell. */
export function platform() {
  if (globalThis.webkit?.messageHandlers?.eidola) return "ios";
  if (globalThis.eidolaShellAndroid) return "android";
  return undefined;
}

/** iOS: posts one request; the promise resolves to its reply's text. */
export function postIos(text) {
  return globalThis.webkit.messageHandlers.eidola.postMessage(text);
}

/** Android: asks the shell for its port; every reply's text goes to `onReply`. */
export function openAndroid(onReply) {
  addEventListener("message", function once(e) {
    if (e.data === "eidola-shell" && e.ports.length === 1) {
      removeEventListener("message", once);
      port = e.ports[0];
      port.onmessage = (m) => onReply(m.data);
      for (const text of queued) port.postMessage(text);
      queued = null;
    }
  });
  globalThis.eidolaShellAndroid.open();
}

/** Android: posts one request (held until the port arrives). */
export function postAndroid(text) {
  if (port) port.postMessage(text);
  else queued.push(text);
}

/** The webview's preferred languages (they follow the device's), as JSON. */
export function preferredLanguages() {
  const list = globalThis.navigator?.languages ?? [globalThis.navigator?.language];
  return JSON.stringify((list ?? []).filter((l) => typeof l === "string"));
}
