// Exercises every call of mobile/shell-contract.md against the shell this page runs in.
//
// The bundle-store checks span several document loads, so the run is phased: each phase
// records its results and the next phase in a file (through the file API), then reloads.
// A: hello, serving, files, pipe, staging; activates "good" (a copy of this page).
// B: on trial in "good": confirm; stage "bad" (reloads without confirming); activate it.
// C: "bad" failed its trial: back on "good"; activate the embedded bundle.
// D: embedded; activate "bad" again, with no confirmed version behind it.
// E: "bad" failed: back on embedded; clean up; report.
// After E, a cold relaunch checks that a synced file survived the process.

import { b64, call, available } from "./shell.js";

const OWN_FILES = ["index.html", "probe.wasm", "shell.js", "style.css", "test.js"];
const PROGRESS = "contract-progress.json";
const REPORT = "contract-report.json";
const KEEP = "contract-keep.bin";
const ECHO = [["127.0.0.1", 7777], ["10.0.2.2", 7777]];
const CSP =
  "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; " +
  "img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; object-src 'none'; " +
  "base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

const enc = new TextEncoder();
const dec = new TextDecoder();
const B = (s) => b64.encode(typeof s === "string" ? enc.encode(s) : s);
const S = (d) => dec.decode(b64.decode(d));
const $ = (id) => document.getElementById(id);

let results = [];
function show(text) {
  $("log").textContent += text + "\n";
  console.log(text);
}
async function test(name, fn) {
  try {
    await fn();
    results.push({ name, ok: true });
    show(`ok    ${name}`);
  } catch (e) {
    results.push({ name, ok: false, error: String(e?.message ?? e) });
    show(`FAIL  ${name}: ${e?.message ?? e}`);
  }
}
/** JSON with object keys sorted, so member order never matters. */
function canon(v) {
  if (Array.isArray(v)) return `[${v.map(canon).join(",")}]`;
  if (v && typeof v === "object") return `{${Object.keys(v).sort().map((k) => `${JSON.stringify(k)}:${canon(v[k])}`).join(",")}}`;
  return JSON.stringify(v);
}
function eq(actual, expected, what = "value") {
  const a = canon(actual);
  const b = canon(expected);
  if (a !== b) throw new Error(`${what}: expected ${b}, got ${a}`);
}
function ok(cond, what) {
  if (!cond) throw new Error(what);
}
async function rejects(promise, code) {
  try {
    await promise;
  } catch (e) {
    if (e.code === code) return;
    throw new Error(`expected ${code}, got ${e.code}: ${e.message}`);
  }
  throw new Error(`expected ${code}, got success`);
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function random(n) {
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i += 65536) crypto.getRandomValues(out.subarray(i, Math.min(n, i + 65536)));
  return out;
}
async function sha256(bytes) {
  const h = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...h].map((b) => b.toString(16).padStart(2, "0")).join("");
}
/** The contract's bundle digest, computed here independently of the shell. */
async function bundleDigest(files) {
  const paths = [...files.keys()].sort();
  let manifest = "";
  for (const p of paths) manifest += `${await sha256(files.get(p))}  ${p}\n`;
  return sha256(enc.encode(manifest));
}
async function ownFiles() {
  const files = new Map();
  for (const p of OWN_FILES) files.set(p, new Uint8Array(await (await fetch(p)).arrayBuffer()));
  return files;
}
async function stage(version, files) {
  await call("bundle.begin", { version });
  for (const [path, bytes] of files) {
    // Two chunks or more each, at offsets, to exercise partial writes.
    const step = Math.max(1, Math.min(1 << 20, Math.ceil(bytes.length / 2)));
    for (let at = 0; at < bytes.length || at === 0; at += step) {
      await call("bundle.write", { version, path, offset: at, data: B(bytes.subarray(at, at + step)) });
      if (bytes.length === 0) break;
    }
  }
  return call("bundle.seal", { version });
}
async function readFile(name) {
  const r = await call("files.read", { name });
  return r.data === null ? null : b64.decode(r.data);
}
async function save(progress) {
  await call("files.write", { name: PROGRESS, offset: 0, data: B(JSON.stringify(progress)), sync: true });
  await call("files.truncate", { name: PROGRESS, size: enc.encode(JSON.stringify(progress)).length, sync: true });
}

// ---------------------------------------------------------------- phase A

async function checkHello() {
  await test("shell.hello: contract 2, platform, methods", async () => {
    const h = await call("shell.hello");
    eq(h.contract, 2, "contract");
    ok(h.platform === "ios" || h.platform === "android", `platform ${h.platform}`);
    ok(typeof h.shell === "string" && h.shell.length > 0, "shell version");
    const expected = [
      "bundle.activate", "bundle.begin", "bundle.confirm", "bundle.remove", "bundle.seal", "bundle.state",
      "bundle.write", "files.list", "files.read", "files.remove", "files.rename", "files.sync",
      "files.truncate", "files.write", "shell.hello", "shell.openExternal", "tcp.close", "tcp.connect", "tcp.receive", "tcp.send",
      "tcp.shutdown",
    ];
    eq(h.methods, expected, "methods");
  });
  await test("unknown method: unknown_method", () => rejects(call("shell.nope"), "unknown_method"));
}

async function checkServing(platform) {
  await test("origin and secure context", () => {
    eq(location.origin, platform === "ios" ? "eidola://app" : "https://app.eidola.invalid", "origin");
    ok(isSecureContext, "secure context");
    ok(crypto.subtle, "crypto.subtle");
  });
  await test("index.html: type, CSP, cache, nosniff", async () => {
    const r = await fetch("index.html");
    eq(r.status, 200, "status");
    eq(r.headers.get("content-type"), "text/html; charset=utf-8", "content-type");
    eq(r.headers.get("content-security-policy"), CSP, "csp");
    eq(r.headers.get("cache-control"), "no-store", "cache-control");
    eq(r.headers.get("x-content-type-options"), "nosniff", "nosniff");
  });
  await test("MIME: js, css", async () => {
    eq((await fetch("shell.js")).headers.get("content-type"), "text/javascript; charset=utf-8", "js");
    eq((await fetch("style.css")).headers.get("content-type"), "text/css; charset=utf-8", "css");
  });
  await test("MIME: wasm instantiates by streaming", async () => {
    eq((await fetch("probe.wasm")).headers.get("content-type"), "application/wasm", "wasm");
    const { instance } = await WebAssembly.instantiateStreaming(fetch("probe.wasm"));
    eq(instance.exports.answer(), 42, "export");
  });
  await test("query ignored, root is index.html", async () => {
    eq((await fetch("shell.js?x=1")).status, 200, "query");
    eq(await (await fetch("/")).text(), await (await fetch("/index.html")).text(), "root");
  });
  await test("404: missing, dot segment, encoded slash", async () => {
    eq((await fetch("missing.txt")).status, 404, "missing");
    eq((await fetch("/.hidden")).status, 404, "dotfile");
    eq((await fetch("/a%2Fb")).status, 404, "encoded");
  });
  await test("405: only GET", async () => {
    eq((await fetch("index.html", { method: "POST", body: "x" })).status, 405, "POST");
  });
  await test("other origins refused", async () => {
    let failed = false;
    try {
      await fetch("https://example.com/");
    } catch (_) {
      failed = true;
    }
    ok(failed, "fetch to another origin succeeded");
  });
  await test("digest of the embedded bundle matches the contract's definition", async () => {
    const st = await call("bundle.state");
    eq(st.serving.version, "embedded", "serving");
    eq(st.serving.digest, await bundleDigest(await ownFiles()), "digest");
  });
}

async function checkFiles() {
  const name = "contract-a.txt";
  await call("files.remove", { name });
  await call("files.remove", { name: "contract-b.txt" });
  await test("files: write at offset, read", async () => {
    eq(await call("files.write", { name, offset: 0, data: B("hello") }), { size: 5 }, "size");
    eq(S((await call("files.read", { name })).data), "hello", "read");
  });
  await test("files: append", async () => {
    eq(await call("files.write", { name, append: true, data: B(" world") }), { size: 11 }, "size");
    eq(S((await call("files.read", { name })).data), "hello world", "read");
  });
  await test("files: write past the end zero-fills", async () => {
    eq(await call("files.write", { name, offset: 20, data: B("!") }), { size: 21 }, "size");
    const d = await readFile(name);
    eq([...d.subarray(11, 20)], new Array(9).fill(0), "gap");
    eq(d[20], 33, "last");
  });
  await test("files: truncate shrinks, grows, creates", async () => {
    await call("files.truncate", { name, size: 5 });
    eq(S((await call("files.read", { name })).data), "hello", "shrunk");
    await call("files.truncate", { name, size: 8, sync: true });
    eq([...(await readFile(name))], [104, 101, 108, 108, 111, 0, 0, 0], "grown");
    await call("files.remove", { name: "contract-t.bin" });
    await call("files.truncate", { name: "contract-t.bin", size: 3 });
    eq([...(await readFile("contract-t.bin"))], [0, 0, 0], "created");
    await call("files.remove", { name: "contract-t.bin" });
  });
  await test("files: sync, durable write, sync of a missing file", async () => {
    await call("files.write", { name, offset: 0, data: B("HELLO"), sync: true });
    await call("files.sync", { name });
    await call("files.sync", { name: "contract-missing" });
    eq(S((await call("files.read", { name })).data).slice(0, 5), "HELLO", "read");
  });
  await test("files: rename replaces atomically; missing source is not_found", async () => {
    await call("files.write", { name: "contract-b.txt", offset: 0, data: B("old") });
    await call("files.rename", { from: name, to: "contract-b.txt" });
    eq(await call("files.read", { name }), { data: null }, "source gone");
    eq(S((await call("files.read", { name: "contract-b.txt" })).data).slice(0, 5), "HELLO", "replaced");
    await rejects(call("files.rename", { from: name, to: "contract-c.txt" }), "not_found");
  });
  await test("files: list is sorted and holds the file", async () => {
    const { names } = await call("files.list");
    ok(names.includes("contract-b.txt"), "listed");
    eq(names, [...names].sort(), "sorted");
  });
  await test("files: remove, and remove of a missing file", async () => {
    await call("files.remove", { name: "contract-b.txt" });
    await call("files.remove", { name: "contract-b.txt" });
    eq(await call("files.read", { name: "contract-b.txt" }), { data: null }, "gone");
  });
  await test("files: invalid names are bad_request", async () => {
    for (const bad of ["", ".hidden", "../x", "a/b", "a b", "é", "x".repeat(256)]) {
      await rejects(call("files.read", { name: bad }), "bad_request");
    }
    await call("files.read", { name: "x".repeat(255) });
  });
  await test("files: offset xor append; base64 checked", async () => {
    await rejects(call("files.write", { name, data: B("x") }), "bad_request");
    await rejects(call("files.write", { name, offset: 0, append: true, data: B("x") }), "bad_request");
    await rejects(call("files.write", { name, offset: 0, data: "!!notbase64" }), "bad_request");
    await rejects(call("files.write", { name, offset: -1, data: B("x") }), "bad_request");
    await rejects(call("files.write", { name, offset: 1.5, data: B("x") }), "bad_request");
  });
  await test("files: 2 MiB round trip", async () => {
    const big = random(2 << 20);
    await call("files.write", { name, offset: 0, data: B(big) });
    await call("files.truncate", { name, size: big.length });
    eq(await sha256(await readFile(name)), await sha256(big), "bytes");
  });
  await test("files: requests apply in arrival order", async () => {
    await call("files.truncate", { name, size: 0 });
    const all = [];
    for (let i = 0; i < 40; i++) all.push(call("files.write", { name, append: true, data: B(`${i},`) }));
    await Promise.all(all);
    eq(S((await call("files.read", { name })).data), [...Array(40).keys()].map((i) => `${i},`).join(""), "order");
    await call("files.remove", { name });
  });
}

/** The flushes `fn` made, from the debug builds' recorder; null in a release build. */
async function flushes(fn) {
  try {
    await call("debug.syncLog");
  } catch (e) {
    if (e.code === "unknown_method") return null;
    throw e;
  }
  await fn();
  return (await call("debug.syncLog")).syncs;
}

async function checkDurability() {
  const d = "contract-d.txt";
  const f = (name) => `files/${name}`;
  await call("files.remove", { name: d });
  await call("files.remove", { name: "contract-e.txt" });
  const cases = [
    ["an unsynced write flushes nothing", () => call("files.write", { name: d, offset: 0, data: B("x") }), []],
    // The name was created by the unsynced write above, so this call did not create it.
    ["a synced write flushes the file, then its directory", () => call("files.write", { name: d, append: true, data: B("y"), sync: true }), [f(d), "files"]],
    ["a synced truncate flushes the file, then its directory", () => call("files.truncate", { name: d, size: 1, sync: true }), [f(d), "files"]],
    ["sync flushes the file, then its directory", () => call("files.sync", { name: d }), [f(d), "files"]],
    ["sync of a missing file flushes the directory", () => call("files.sync", { name: "contract-missing" }), ["files"]],
    ["rename flushes the source's contents, then the directory", async () => {
      await call("files.write", { name: "contract-e.txt", offset: 0, data: B("unsynced") });
      await call("files.rename", { from: "contract-e.txt", to: d });
    }, [f("contract-e.txt"), "files"]],
    ["remove flushes the directory", () => call("files.remove", { name: d }), ["files"]],
  ];
  for (const [name, fn, want] of cases) {
    await test(`durability: ${name}`, async () => {
      const got = await flushes(fn);
      if (got === null) return show("      (skipped: a release build records no flushes)");
      eq(got, want, "flushes");
    });
  }
  await test("durability: seal flushes the version, its name, then the state", async () => {
    await call("bundle.begin", { version: "durable" });
    await call("bundle.write", { version: "durable", path: "index.html", offset: 0, data: B("<p>") });
    await call("bundle.write", { version: "durable", path: "a/b.txt", offset: 0, data: B("b") });
    const got = await flushes(() => call("bundle.seal", { version: "durable" }));
    await call("bundle.remove", { version: "durable" });
    if (got === null) return show("      (skipped: a release build records no flushes)");
    const v = "bundles/versions/durable";
    eq([...got.slice(0, 4)].sort(), [`${v}`, `${v}/a`, `${v}/a/b.txt`, `${v}/index.html`], "the version's tree");
    eq(got.slice(4), ["bundles/versions", "bundles/state.json.tmp", "bundles"], "then its name, then the state");
  });
}

let nextSocket = 100;
async function connectEcho() {
  // A device reaches the echo server over the LAN: `host:port` in this file, put there by hand.
  const lan = await readFile("contract-echo.txt");
  const where = lan ? [dec.decode(lan).trim().split(":")].map(([h, p]) => [h, Number(p)]) : [];
  for (const [host, port] of [...where, ...ECHO]) {
    const socket = nextSocket++;
    try {
      await call("tcp.connect", { socket, host, port });
      return { socket, host, port };
    } catch (e) {
      if (e.code !== "connect_failed") throw e;
    }
  }
  throw new Error("no echo server at " + [...where, ...ECHO].map((e) => e.join(":")).join(" or "));
}
async function receiveExactly(socket, n) {
  const parts = [];
  let got = 0;
  while (got < n) {
    const r = await call("tcp.receive", { socket });
    if (r.eof) throw new Error(`eof after ${got} of ${n} bytes`);
    const d = b64.decode(r.data);
    parts.push(d);
    got += d.length;
  }
  const out = new Uint8Array(got);
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

async function checkPipe() {
  let echo = null;
  await test("tcp: connect to the echo server", async () => {
    echo = await connectEcho();
  });
  if (!echo) return;
  const { socket, host, port } = echo;
  show(`      (echo server at ${host}:${port})`);
  await test("tcp: send and receive", async () => {
    await call("tcp.send", { socket, data: B("hello") });
    eq(dec.decode(await receiveExactly(socket, 5)), "hello", "echo");
  });
  await test("tcp: receive honours max", async () => {
    await call("tcp.send", { socket, data: B("abcdef") });
    await sleep(100);
    const r = await call("tcp.receive", { socket, max: 3 });
    const first = b64.decode(r.data);
    ok(first.length >= 1 && first.length <= 3, `got ${first.length} bytes`);
    const rest = await receiveExactly(socket, 6 - first.length);
    eq(dec.decode(first) + dec.decode(rest), "abcdef", "bytes");
    await rejects(call("tcp.receive", { socket, max: 0 }), "bad_request");
  });
  await test("tcp: 1 MiB both ways, sends and receives interleaved", async () => {
    const big = random(1 << 20);
    const sending = (async () => {
      for (let at = 0; at < big.length; at += 1 << 18) {
        await call("tcp.send", { socket, data: B(big.subarray(at, at + (1 << 18))) });
      }
    })();
    const back = await receiveExactly(socket, big.length);
    await sending;
    eq(await sha256(back), await sha256(big), "bytes");
  });
  await test("tcp: one receive at a time", async () => {
    const first = call("tcp.receive", { socket });
    await rejects(call("tcp.receive", { socket }), "invalid_state");
    await call("tcp.send", { socket, data: B("z") });
    eq(S((await first).data), "z", "first receive");
  });
  await test("tcp: shutdown, then the peer's reply and eof", async () => {
    await call("tcp.shutdown", { socket });
    await rejects(call("tcp.send", { socket, data: B("late") }), "invalid_state");
    let text = "";
    for (;;) {
      const r = await call("tcp.receive", { socket });
      if (r.eof) break;
      text += S(r.data);
    }
    eq(text, `bye ${5 + 6 + (1 << 20) + 1}\n`, "reply");
    eq(await call("tcp.receive", { socket }), { eof: true }, "eof again");
  });
  await test("tcp: close forgets the id; closing again is fine", async () => {
    await call("tcp.close", { socket });
    await call("tcp.close", { socket });
    await rejects(call("tcp.send", { socket, data: B("x") }), "not_found");
  });
  await test("tcp: close fails a pending receive with closed", async () => {
    const s = nextSocket++;
    await call("tcp.connect", { socket: s, host, port });
    const pending = call("tcp.receive", { socket: s });
    await sleep(50);
    await call("tcp.close", { socket: s });
    await rejects(pending, "closed");
  });
  await test("tcp: two sockets at once", async () => {
    const a = nextSocket++;
    const b = nextSocket++;
    await Promise.all([call("tcp.connect", { socket: a, host, port }), call("tcp.connect", { socket: b, host, port })]);
    await Promise.all([call("tcp.send", { socket: a, data: B("aaaa") }), call("tcp.send", { socket: b, data: B("bb") })]);
    eq(dec.decode(await receiveExactly(b, 2)), "bb", "b");
    eq(dec.decode(await receiveExactly(a, 4)), "aaaa", "a");
    await call("tcp.close", { socket: a });
    await call("tcp.close", { socket: b });
  });
  await test("tcp: id in use is invalid_state", async () => {
    const s = nextSocket++;
    await call("tcp.connect", { socket: s, host, port });
    await rejects(call("tcp.connect", { socket: s, host, port }), "invalid_state");
    await call("tcp.close", { socket: s });
  });
  await test("tcp: refused connect is connect_failed", () =>
    rejects(call("tcp.connect", { socket: nextSocket++, host, port: port + 1 }), "connect_failed"));
  await test("tcp: closing a pending connect fails it with closed", async () => {
    const s = nextSocket++;
    const pending = call("tcp.connect", { socket: s, host: "192.0.2.1", port: 9 });
    await call("tcp.close", { socket: s });
    await rejects(pending, "closed");
  });
  await test("tcp: bad parameters", async () => {
    await rejects(call("tcp.connect", { socket: nextSocket++, host, port: 0 }), "bad_request");
    await rejects(call("tcp.connect", { socket: nextSocket++, host, port: 70000 }), "bad_request");
    await rejects(call("tcp.connect", { socket: nextSocket++, host: "", port }), "bad_request");
    await rejects(call("tcp.receive", { socket: 999999 }), "not_found");
  });
}

const BAD = new Map([
  ["index.html", enc.encode('<!doctype html><meta charset="utf-8"><title>bad</title><script src="bad.js"></script>\n')],
  ["bad.js", enc.encode("// A bundle that never confirms: it reloads, as a crash and relaunch would.\nsetTimeout(() => location.reload(), 300);\n")],
]);

async function phaseA(progress, platform) {
  await checkHello();
  await checkServing(platform);
  await checkFiles();
  await checkDurability();
  await checkPipe();
  await test("bundle: refusals", async () => {
    await rejects(call("bundle.begin", { version: "embedded" }), "invalid_state");
    await rejects(call("bundle.begin", { version: "../x" }), "bad_request");
    await rejects(call("bundle.activate", { version: "nope" }), "not_found");
    await rejects(call("bundle.remove", { version: "embedded" }), "invalid_state");
    await rejects(call("bundle.remove", { version: "nope" }), "not_found");
    await call("bundle.begin", { version: "staging" });
    await rejects(call("bundle.write", { version: "staging", path: "a/../b", offset: 0, data: B("x") }), "bad_request");
    await rejects(call("bundle.write", { version: "staging", path: ".x", offset: 0, data: B("x") }), "bad_request");
    await rejects(call("bundle.write", { version: "nope", path: "x", offset: 0, data: B("x") }), "invalid_state");
    await rejects(call("bundle.activate", { version: "staging" }), "invalid_state");
    const st = await call("bundle.state");
    eq(st.versions.find((v) => v.version === "staging"), { version: "staging", sealed: false, digest: null }, "listed");
    await call("bundle.remove", { version: "staging" });
  });
  await test("bundle: stage a copy of this page; the digest matches", async () => {
    const files = await ownFiles();
    files.set("assets/big.bin", random(3 << 20));
    const sealed = await stage("good", files);
    eq(sealed.digest, await bundleDigest(files), "digest");
    eq(sealed.files, files.size, "files");
    eq(sealed.bytes, [...files.values()].reduce((n, f) => n + f.length, 0), "bytes");
    await rejects(call("bundle.write", { version: "good", path: "x", offset: 0, data: B("x") }), "invalid_state");
    await rejects(call("bundle.begin", { version: "good" }), "invalid_state");
    progress.goodDigest = sealed.digest;
  });
  await test("files: a synced file to outlive reloads and the process", async () => {
    const keep = random(100000);
    await call("files.write", { name: KEEP, offset: 0, data: B(keep), sync: true });
    await call("files.truncate", { name: KEEP, size: keep.length, sync: true });
    progress.keep = await sha256(keep);
  });
  await test("bundle: activate is provisional", async () => {
    const st = await call("bundle.activate", { version: "good" });
    eq([st.active, st.confirmed, st.previous, st.serving.version], ["good", false, "embedded", "embedded"], "state");
  });
  progress.phase = "B";
}

async function phaseB(progress) {
  await test("trial: the new bundle boots unconfirmed", async () => {
    const st = await call("bundle.state");
    eq([st.serving.version, st.serving.confirmed, st.active, st.confirmed], ["good", false, "good", false], "state");
    eq(st.serving.digest, progress.goodDigest, "digest from the served bytes");
  });
  await test("files: the synced file survived the reload", async () => {
    eq(await sha256(await readFile(KEEP)), progress.keep, "bytes");
  });
  await test("trial: confirm", async () => {
    const st = await call("bundle.confirm");
    eq([st.serving.confirmed, st.active, st.confirmed, st.previous], [true, "good", true, "embedded"], "state");
    eq((await call("bundle.confirm")).confirmed, true, "again");
  });
  await test("bundle: stage and activate one that never confirms", async () => {
    const sealed = await stage("bad", BAD);
    eq(sealed.digest, await bundleDigest(BAD), "digest");
    const st = await call("bundle.activate", { version: "bad" });
    eq([st.active, st.confirmed, st.previous], ["bad", false, "good"], "state");
    await rejects(call("bundle.remove", { version: "good" }), "invalid_state");
  });
  progress.phase = "C";
}

async function phaseC(progress) {
  await test("fallback: a failed trial returns to the previous confirmed bundle", async () => {
    const st = await call("bundle.state");
    eq([st.serving.version, st.serving.confirmed, st.active, st.confirmed, st.previous, st.failed],
      ["good", true, "good", true, "embedded", "bad"], "state");
    eq(st.versions.map((v) => [v.version, v.sealed]), [["bad", true], ["good", true]], "versions");
  });
  await test("bundle: activating embedded", async () => {
    const st = await call("bundle.activate", { version: "embedded" });
    eq([st.active, st.confirmed, st.previous], ["embedded", true, "embedded"], "state");
    await rejects(call("bundle.remove", { version: "good" }), "invalid_state");
  });
  progress.phase = "D";
}

async function phaseD(progress) {
  await test("embedded again after activating it", async () => {
    const st = await call("bundle.state");
    eq([st.serving.version, st.active, st.failed], ["embedded", "embedded", "bad"], "state");
  });
  await test("bundle: activate a bad one with no confirmed version behind it", async () => {
    await call("bundle.remove", { version: "good" });
    const st = await call("bundle.activate", { version: "bad" });
    eq([st.active, st.confirmed, st.previous], ["bad", false, "embedded"], "state");
  });
  progress.phase = "E";
}

async function phaseE(progress) {
  await test("fallback: a failed trial with nothing confirmed returns to embedded", async () => {
    const st = await call("bundle.state");
    eq([st.serving.version, st.active, st.confirmed, st.previous, st.failed], ["embedded", "embedded", true, "embedded", "bad"], "state");
  });
  await test("bundle: remove the rest", async () => {
    await call("bundle.remove", { version: "bad" });
    eq((await call("bundle.state")).versions, [], "versions");
  });
  await test("openExternal: refuses anything but an absolute https URL", async () => {
    for (const url of [
      "http://example.com/", "javascript:alert(1)", "data:text/html,x", "/relative", "example.com",
      "https:///no-host", "https://exa mple.com/", "file:///etc/hosts", "eidola://app/", "",
    ]) {
      await rejects(call("shell.openExternal", { url }), "bad_request");
    }
    await rejects(call("shell.openExternal", {}), "bad_request");
  });
  // Last, because the system browser comes to the front (the harness brings this app back).
  await test("openExternal: an https URL goes to the system browser; this page stays", async () => {
    const here = location.href;
    globalThis.contractMarker = true;
    await call("shell.openExternal", { url: "https://example.com/" });
    eq(location.href, here, "location");
    ok(globalThis.contractMarker, "the same document");
  });
  progress.phase = "done";
}

async function finish(platform, progress) {
  const failed = results.filter((r) => !r.ok);
  const summary = `contract: ${failed.length ? "FAIL" : "PASS"} ${results.length - failed.length}/${results.length}`;
  await call("files.write", {
    name: REPORT, offset: 0, data: B(JSON.stringify({ platform, summary, results }, null, 1)), sync: true,
  });
  progress.summary = summary;
  await save(progress);
  return summary;
}

async function main() {
  if (!available()) {
    $("summary").textContent = "contract: no shell (open this page in a shell build)";
    return;
  }
  const { platform } = await call("shell.hello");
  const st = await call("bundle.state");
  $("digest").textContent = `serving ${st.serving.version}, digest ${st.serving.digest}`;
  const raw = await readFile(PROGRESS);
  // An empty progress file starts a fresh run (how a device session restarts one).
  const progress = raw?.length ? JSON.parse(dec.decode(raw)) : { phase: "A", results: [] };
  results = progress.results;
  if (progress.phase === "done") {
    // A cold relaunch after a complete run.
    const keep = await readFile(KEEP);
    const survived = keep !== null && (await sha256(keep)) === progress.keep;
    $("summary").textContent = progress.summary;
    $("restart").textContent = `restart: ${survived ? "PASS" : "FAIL"} (synced file ${survived ? "intact" : "lost"})`;
    show(results.map((r) => `${r.ok ? "ok   " : "FAIL "} ${r.name}`).join("\n"));
    const report = JSON.parse(dec.decode(await readFile(REPORT)));
    report.restart = $("restart").textContent;
    await call("files.write", { name: REPORT, offset: 0, data: B(JSON.stringify(report, null, 1)), sync: true });
    await call("files.remove", { name: KEEP });
    await call("files.remove", { name: PROGRESS });
    return;
  }
  $("summary").textContent = `contract: running phase ${progress.phase}…`;
  show(`phase ${progress.phase} (${platform}, serving ${st.serving.version})`);
  const phase = { A: phaseA, B: phaseB, C: phaseC, D: phaseD, E: phaseE }[progress.phase];
  try {
    await phase(progress, platform);
  } catch (e) {
    results.push({ name: `phase ${progress.phase}`, ok: false, error: String(e?.message ?? e) });
    progress.phase = "done";
  }
  if (progress.phase === "done") {
    $("summary").textContent = await finish(platform, progress);
  } else {
    await save(progress);
    location.reload();
  }
}

main().catch((e) => {
  $("summary").textContent = `contract: FAIL (${e?.code ?? ""} ${e?.message ?? e})`;
});
