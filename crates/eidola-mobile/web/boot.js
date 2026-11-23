// Loads the app's wasm and boots it on the shell (src/web.rs → start).
import init, { start } from "./eidola_mobile.js";

const status = document.getElementById("status");
try {
  await init();
  status.textContent = await start();
} catch (e) {
  status.textContent = String(e?.message ?? e);
}
