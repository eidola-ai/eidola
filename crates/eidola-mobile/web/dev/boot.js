// A development bundle's boot (mobile/build-bundle.sh --dev-backend): the same
// app, started with the development server dev-backend.json names.
import init, { startDev } from "./eidola_mobile.js";

try {
  await init();
  const config = await (await fetch("dev-backend.json")).text();
  startDev(config);
} catch (e) {
  document.getElementById("status").textContent = String(e?.message ?? e);
}
