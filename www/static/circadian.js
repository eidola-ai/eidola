// The circadian theme runtime — a port of the app's theme.rs/solar.rs
// resolution logic (crates/eidola-gui/src/). It picks one of six palettes
// (day/night × cool/neutral/warm, defined in the generated circadian.css)
// from the sun's schedule and sets it as data-palette on <html>.
//
// Defaults mirror the app: appearance = auto (light between canonical
// 06:00 and 18:00, dark otherwise), time-of-day tint = on. The footer
// control (a ☀/☾ disclosure — the glyph reflects the mode in use — that
// expands inline to the options) can pin appearance to day/night, or follow
// the OS color scheme ("system"), all persisted in localStorage; there is
// no tint control on the site.
//
// Geography, like the app, is inferred from the IANA timezone — never
// from geolocation. The browser has no tzdb, so zones.js ships a snapshot
// of zone.tab's representative coordinates. Unknown zones degrade to a
// fixed 06:00/18:00 clock, exactly like solar.rs's DayPhases::Clock.
(function () {
  "use strict";

  var STORAGE_KEY = "eidola-appearance";

  function sind(d) {
    return Math.sin((d * Math.PI) / 180);
  }
  function cosd(d) {
    return Math.cos((d * Math.PI) / 180);
  }
  function remEuclid(x, m) {
    return ((x % m) + m) % m;
  }

  // The NOAA sunrise equation, matching solar.rs solar_events(). Returns
  // {kind:"events", sunrise, sunset} in local minutes-since-midnight, or
  // {kind:"polar-day"|"polar-night"|"clock"}.
  function solarEvents(unixSecs, lat, lon, utcOffsetMin) {
    var jdate = unixSecs / 86400 + 2440587.5;
    var n = Math.round(jdate - 2451545.0 + 0.0008);
    var jStar = n - lon / 360;
    var m = remEuclid(357.5291 + 0.98560028 * jStar, 360);
    var c = 1.9148 * sind(m) + 0.02 * sind(2 * m) + 0.0003 * sind(3 * m);
    var lambda = remEuclid(m + c + 180 + 102.9372, 360);
    var jTransit = 2451545.0 + jStar + 0.0053 * sind(m) - 0.0069 * sind(2 * lambda);
    var sinDecl = sind(lambda) * sind(23.4397);
    var cosDecl = Math.sqrt(1 - sinDecl * sinDecl);
    var cosOmega = (sind(-0.833) - sind(lat) * sinDecl) / (cosd(lat) * cosDecl);
    if (cosOmega > 1) return { kind: "polar-night" }; // sun never rises
    if (cosOmega < -1) return { kind: "polar-day" }; // sun never sets
    var omega = (Math.acos(cosOmega) * 180) / Math.PI;
    var toLocalMin = function (j) {
      var unix = (j - 2440587.5) * 86400;
      return remEuclid(unix / 60 + utcOffsetMin, 1440);
    };
    return {
      kind: "events",
      sunrise: toLocalMin(jTransit - omega / 360),
      sunset: toLocalMin(jTransit + omega / 360),
    };
  }

  // Warp wall time so sunrise lands at canonical 06:00 and sunset at
  // 18:00 (theme.rs canonical_hour) — December's sunset character arrives
  // around 16:30 real time, June's around 20:30.
  function canonicalHour(nowMin, phases) {
    if (phases.kind !== "events") return remEuclid(nowMin / 60, 24);
    var dayLen = remEuclid(phases.sunset - phases.sunrise, 1440);
    if (dayLen === 0) return remEuclid(nowMin / 60, 24);
    var sinceRise = remEuclid(nowMin - phases.sunrise, 1440);
    if (sinceRise < dayLen) return 6 + (12 * sinceRise) / dayLen;
    var nightLen = 1440 - dayLen;
    var sinceSet = sinceRise - dayLen;
    return remEuclid(18 + (12 * sinceSet) / nightLen, 24);
  }

  // The character schedule (theme.rs character_for_hour): tinted windows
  // hug the sun's events by ±2 canonical hours.
  function characterForHour(h) {
    if (h >= 4 && h < 8) return "cool"; // dawn + sunrise
    if (h >= 8 && h < 16) return "neutral"; // the long day
    if (h >= 16 && h < 20) return "warm"; // sunset + dusk
    return "neutral"; // the long night
  }

  function phases() {
    var zones = window.EIDOLA_ZONES || {};
    var zone = "";
    try {
      zone = Intl.DateTimeFormat().resolvedOptions().timeZone || "";
    } catch (e) {
      /* fall through to clock */
    }
    var coords = zones[zone];
    if (!coords) return { kind: "clock" };
    var now = new Date();
    return solarEvents(
      now.getTime() / 1000,
      coords[0],
      coords[1],
      -now.getTimezoneOffset()
    );
  }

  function appearance() {
    var stored = null;
    try {
      stored = localStorage.getItem(STORAGE_KEY);
    } catch (e) {
      /* private mode etc. */
    }
    return stored === "day" || stored === "night" || stored === "system"
      ? stored
      : "auto";
  }

  // Whether the OS reports a dark color scheme. Used only in "system" mode.
  function prefersDark() {
    try {
      return !!(
        window.matchMedia &&
        window.matchMedia("(prefers-color-scheme: dark)").matches
      );
    } catch (e) {
      return false;
    }
  }

  function resolve() {
    var now = new Date();
    var nowMin = now.getHours() * 60 + now.getMinutes();
    var ph = phases();
    var canonical = canonicalHour(nowMin, ph);

    var pref = appearance();
    var isDay;
    if (pref === "day") isDay = true;
    else if (pref === "night") isDay = false;
    else if (pref === "system") isDay = !prefersDark();
    else if (ph.kind === "polar-day") isDay = true;
    else if (ph.kind === "polar-night") isDay = false;
    else isDay = canonical >= 6 && canonical < 18;

    var character = characterForHour(canonical);
    var palette = (isDay ? "day" : "night") + "-" + character;

    var root = document.documentElement;
    if (root.dataset.palette !== palette) root.dataset.palette = palette;

    // The collapsed trigger shows the icon of the mode actually in use.
    var summary = document.querySelector(".appearance > summary");
    if (summary) {
      var glyph = isDay ? "☀" : "☾";
      if (summary.textContent !== glyph) summary.textContent = glyph;
    }

    var buttons = document.querySelectorAll(".appearance button");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].setAttribute(
        "aria-pressed",
        buttons[i].dataset.appearance === pref ? "true" : "false"
      );
    }

    // Enable palette crossfades only after the first paint has settled.
    if (!root.classList.contains("circadian-ready")) {
      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          root.classList.add("circadian-ready");
        });
      });
    }
  }

  function wireToggle() {
    var details = document.querySelector("details.appearance");
    var buttons = document.querySelectorAll(".appearance button");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].addEventListener("click", function (ev) {
        var value = ev.currentTarget.dataset.appearance;
        try {
          if (value === "auto") localStorage.removeItem(STORAGE_KEY);
          else localStorage.setItem(STORAGE_KEY, value);
        } catch (e) {
          /* ignore */
        }
        if (details) details.removeAttribute("open"); // collapse after choosing
        resolve();
      });
    }

    if (!details) return;
    // Collapse the disclosure on an outside click or Escape.
    document.addEventListener("click", function (ev) {
      if (details.hasAttribute("open") && !details.contains(ev.target)) {
        details.removeAttribute("open");
      }
    });
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape") details.removeAttribute("open");
    });
  }

  // Re-resolve when the OS color scheme flips, but only while in "system" mode.
  try {
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    var onSchemeChange = function () {
      if (appearance() === "system") resolve();
    };
    if (mq.addEventListener) mq.addEventListener("change", onSchemeChange);
    else if (mq.addListener) mq.addListener(onSchemeChange);
  } catch (e) {
    /* ignore */
  }

  wireToggle();
  resolve();
  // The app re-resolves once a minute; palette only changes on slot
  // boundaries, and resolve() is cheap.
  setInterval(resolve, 60000);
})();
