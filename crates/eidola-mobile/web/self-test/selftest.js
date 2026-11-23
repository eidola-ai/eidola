// The self-test bundle's page (src/selftest.rs): the harness's selftest.json in,
// a report out.
import init, { selfTest } from "./eidola_mobile.js";

// What of the webview's own storage this page can see: null where a store is
// unavailable, else how many items it holds.
function webviewStorage() {
  const count = (name) => {
    try {
      const store = globalThis[name];
      return store ? store.length : null;
    } catch {
      return null;
    }
  };
  return JSON.stringify({ localStorage: count("localStorage"), sessionStorage: count("sessionStorage") });
}

const verdict = document.getElementById("verdict");
try {
  await init();
  const config = await (await fetch("selftest.json")).text();
  const text = await selfTest(config, webviewStorage());
  const report = JSON.parse(text);
  verdict.textContent = `self-test phase ${report.phase}: ${report.pass ? "PASS" : "FAIL"}`;
  document.getElementById("report").textContent = text;
} catch (e) {
  verdict.textContent = `self-test: ERROR ${e?.message ?? e}`;
}
