// The page's own calls (src/web.rs): a reload, and a file of this document's own origin — the
// bundle the shell served. The shell serves nothing else, and the page's policy connects nowhere
// else.

/** Load the document again: the shell serves whichever bundle is now active. */
export function reloadPage() {
  location.reload();
}

/** A bundle file, as text. */
export async function fetchText(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return await response.text();
}

/**
 * `onVisible` runs each time the reader comes back to the app: the page becomes visible again
 * (the app returns to the foreground), or the window regains focus after losing it. On an iPad
 * another app can open in a window over this one, the browser after a checkout among them, and
 * the page then never stops being visible: only its focus says the reader went and came back.
 */
export function watchVisibility(onVisible) {
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") onVisible();
  });
  let blurred = false;
  window.addEventListener("blur", () => {
    blurred = true;
  });
  window.addEventListener("focus", () => {
    if (blurred) {
      blurred = false;
      onVisible();
    }
  });
}
