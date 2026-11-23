// The page's session history, mirroring the app's screen stack (src/ui/nav.rs): one entry per
// screen above the root, so Android's back gesture (which the shell turns into a history step
// while there is one) pops a screen. A step back the app takes itself is made quietly.

let depth = 0;
let quiet = 0;

/** Starts listening; `onBack` runs for a back gesture. After a reload, steps back to the root. */
export function navInit(onBack) {
  addEventListener("popstate", () => {
    if (quiet > 0) {
      quiet -= 1;
      return;
    }
    if (depth > 0) depth -= 1;
    onBack();
  });
  const stale = history.state?.eidolaDepth ?? 0;
  if (stale > 0) {
    quiet += 1;
    history.go(-stale);
  }
}

/** A screen was pushed. */
export function navPush() {
  depth += 1;
  history.pushState({ eidolaDepth: depth }, "");
}

/** The app popped a screen itself. */
export function navPopped() {
  if (depth > 0) {
    depth -= 1;
    quiet += 1;
    history.back();
  }
}

/** The stack went back to its root. */
export function navReset() {
  if (depth > 0) {
    quiet += 1;
    history.go(-depth);
    depth = 0;
  }
}
