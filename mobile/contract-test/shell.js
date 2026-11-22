// Reference bridge to the native shell (mobile/shell-contract.md → The bridge): one
// `call(method, params)` over whichever transport the platform has.

let next = 1;
let transport = null;

function ios() {
  const handler = globalThis.webkit?.messageHandlers?.eidola;
  if (!handler) return null;
  return (request) => handler.postMessage(JSON.stringify(request)).then((text) => JSON.parse(text));
}

function android() {
  if (!globalThis.eidolaShellAndroid) return null;
  const pending = new Map();
  const port = new Promise((resolve) => {
    addEventListener("message", function once(e) {
      if (e.data === "eidola-shell" && e.ports.length === 1) {
        removeEventListener("message", once);
        e.ports[0].onmessage = (m) => {
          const reply = JSON.parse(m.data);
          pending.get(reply.id)?.(reply);
          pending.delete(reply.id);
        };
        resolve(e.ports[0]);
      }
    });
    globalThis.eidolaShellAndroid.open();
  });
  return (request) =>
    port.then(
      (p) =>
        new Promise((resolve) => {
          pending.set(request.id, resolve);
          p.postMessage(JSON.stringify(request));
        }),
    );
}

/** The platform's transport, or null outside a shell. */
export function available() {
  transport ??= ios() ?? android();
  return transport !== null;
}

/** Calls the shell; resolves to the result, or rejects with an Error carrying `code`. */
export async function call(method, params = {}) {
  if (!available()) throw Object.assign(new Error("no shell"), { code: "no_shell" });
  const reply = await transport({ id: next++, method, params });
  if (reply.ok) return reply.result;
  throw Object.assign(new Error(`${method}: ${reply.error.code}: ${reply.error.message}`), { code: reply.error.code });
}

export const b64 = {
  encode(bytes) {
    let s = "";
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    return btoa(s);
  },
  decode(text) {
    const s = atob(text);
    const out = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
    return out;
  },
};
