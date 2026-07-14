// Scroll-spy for the in-page table of contents (.toc) — the right-hand
// rail on docs and posts, a cousin of the app's minimap. The entry whose
// section is currently being read carries aria-current; entries already
// read are marked .passed, so the rail's filled edge doubles as a reading
// progress indicator.
(function () {
  "use strict";

  var toc = document.querySelector(".toc");
  if (!toc) return;

  var links = Array.prototype.slice.call(toc.querySelectorAll("a[href^='#']"));
  var targets = links.map(function (link) {
    return document.getElementById(
      decodeURIComponent(link.getAttribute("href").slice(1))
    );
  });

  function update() {
    // The active section is the last heading at or above the reading
    // line (a little below the chrome); at the very end of the page,
    // snap to the last entry so it can always be reached.
    var active = -1;
    for (var i = 0; i < targets.length; i++) {
      if (targets[i] && targets[i].getBoundingClientRect().top <= 96) {
        active = i;
      }
    }
    if (
      window.innerHeight + window.scrollY >=
      document.documentElement.scrollHeight - 2
    ) {
      active = targets.length - 1;
    }
    for (var j = 0; j < links.length; j++) {
      links[j].classList.toggle("passed", j < active);
      if (j === active) {
        links[j].setAttribute("aria-current", "true");
      } else {
        links[j].removeAttribute("aria-current");
      }
    }
  }

  var scheduled = false;
  function schedule() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(function () {
      scheduled = false;
      update();
    });
  }

  window.addEventListener("scroll", schedule, { passive: true });
  window.addEventListener("resize", schedule, { passive: true });
  update();
})();
