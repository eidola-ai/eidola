// Development bundles only (feature `dev`): mirror the frame's probes
// (src/ui/probe.rs) into the page as invisible, labelled elements that let
// touches through, so UI tests find them in the webview's own accessibility
// tree by name and tap their centres onto the canvas beneath.

let layer = null;
const nodes = new Map();

/** Replace the mirror with the entries in `json` ([{name, role, x, y, w, h}]). */
export function syncProbes(json) {
  if (!layer) {
    layer = document.createElement("div");
    layer.id = "eidola-probes";
    layer.style.cssText = "position:fixed;inset:0;pointer-events:none;z-index:2147483647";
    document.body.appendChild(layer);
  }
  const seen = new Set();
  for (const e of JSON.parse(json)) {
    seen.add(e.name);
    let node = nodes.get(e.name);
    if (!node) {
      node = document.createElement("div");
      node.setAttribute("aria-label", e.name);
      node.style.cssText = "position:absolute;pointer-events:none";
      layer.appendChild(node);
      nodes.set(e.name, node);
    }
    node.setAttribute("role", e.role === "TextInput" ? "textbox" : e.role === "Button" ? "button" : "group");
    node.dataset.label = e.label;
    node.style.left = `${e.x}px`;
    node.style.top = `${e.y}px`;
    node.style.width = `${Math.max(e.w, 1)}px`;
    node.style.height = `${Math.max(e.h, 1)}px`;
  }
  for (const [name, node] of nodes) {
    if (!seen.has(name)) {
      node.remove();
      nodes.delete(name);
    }
  }
}
