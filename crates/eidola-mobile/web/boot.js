// Loads the app's wasm and starts it on the shell (src/web.rs → start): the
// app draws its own window, boots behind it, and confirms the bundle once it
// has drawn. Only a failure to load or start the wasm is reported here.
import init, { start } from "./eidola_mobile.js";

try {
  await init();
  start();
} catch (e) {
  document.getElementById("status").textContent = String(e?.message ?? e);
}
