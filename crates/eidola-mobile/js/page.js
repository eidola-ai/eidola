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

/** `onVisible` runs each time the page becomes visible again (the app returns to the foreground). */
export function watchVisibility(onVisible) {
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") onVisible();
  });
}
