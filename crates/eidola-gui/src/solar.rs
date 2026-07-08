//! Solar events from the system timezone — the geographic half of the
//! circadian theme.
//!
//! An IANA zone ID isn't just an offset rule: it names a real place, and
//! the tz database that ships with every OS carries coordinates for each
//! zone (`zone1970.tab` / `zone.tab` — on macOS under `/usr/share/zoneinfo`,
//! a symlink into `/var/db/timezone`). So without any location permission
//! or network access we can read the zone name (`TZ` or the
//! `/etc/localtime` symlink), look up its representative coordinates, and
//! run the standard sunrise equation.
//!
//! Accuracy is bounded by geography, not the math: longitude is pinned to
//! within the zone (±~30 min of solar time), latitude by the zone's
//! representative city (large countries can be several degrees off), so
//! sunrise/sunset land within roughly ±30–60 minutes for most users —
//! far closer than the fixed 06:00/18:00 clock schedule, whose seasonal
//! error at mid-latitudes exceeds four hours. Known coarse spots: zones
//! politically far from solar time (western China on `Asia/Shanghai`),
//! geography-free zones (`UTC`, `Etc/*` — no coordinates, callers fall
//! back to the clock schedule), and polar latitudes, where "no sunrise
//! today" is a real result ([`SolarEvents::PolarDay`] / [`PolarNight`]).
//!
//! Everything here is pure std + libc (no chrono, no network); the theme's
//! clock task calls it once a minute, which is far below any caching
//! concern except the `.tab` file scan, which `theme.rs` memoizes per zone
//! name.
//!
//! [`PolarNight`]: SolarEvents::PolarNight

use std::path::Path;

/// Today's solar events at a location, in the observer's local wall time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SolarEvents {
    /// The sun rises and sets today.
    Normal {
        /// Sunrise, minutes since local midnight (0..1440).
        sunrise: f32,
        /// Sunset, minutes since local midnight (0..1440).
        sunset: f32,
    },
    /// The sun never sets today (polar day).
    PolarDay,
    /// The sun never rises today (polar night).
    PolarNight,
}

/// The system's IANA zone name (e.g. `America/Los_Angeles`), from `$TZ`
/// when it carries one, otherwise from the `/etc/localtime` symlink.
/// `None` for fixed-offset/POSIX specs and unlinked setups.
pub fn system_zone_name() -> Option<String> {
    if let Ok(tz) = std::env::var("TZ") {
        let tz = tz.trim().trim_start_matches(':');
        if tz.starts_with('/') {
            if let Some(zone) = zone_from_path(tz) {
                return Some(zone);
            }
        } else if tz.contains('/') {
            return Some(tz.to_string());
        }
        // A POSIX spec like `PST8PDT` carries no geography; fall through to
        // the symlink, which may still name a real zone.
    }
    let target = std::fs::read_link("/etc/localtime").ok()?;
    zone_from_path(target.to_str()?)
}

/// Extract `Area/City` from a zoneinfo path such as
/// `/var/db/timezone/zoneinfo/America/Los_Angeles` or
/// `/usr/share/zoneinfo.default/Europe/Paris`.
fn zone_from_path(path: &str) -> Option<String> {
    let idx = path.rfind("/zoneinfo")?;
    let rest = &path[idx + "/zoneinfo".len()..];
    // Skip any suffix on the zoneinfo dir itself (`.default`, versioned
    // dirs), then the separating slash.
    let rest = rest
        .trim_start_matches(|c| c != '/')
        .trim_start_matches('/');
    (!rest.is_empty()).then(|| rest.to_string())
}

/// Representative coordinates (degrees north, degrees east) for an IANA
/// zone, from the OS's tzdb tables. `None` when the zone isn't listed
/// (fixed-offset zones, backward-compat links) or no table is readable.
pub fn zone_coordinates(zone: &str) -> Option<(f64, f64)> {
    // zone1970.tab is preferred where it exists (Linux); macOS ships only
    // zone.tab. Both use the same `CC<tab>ISO6709<tab>ZONE` line format.
    const CANDIDATES: &[&str] = &[
        "/usr/share/zoneinfo/zone1970.tab",
        "/usr/share/zoneinfo/zone.tab",
        "/var/db/timezone/zoneinfo/zone1970.tab",
        "/var/db/timezone/zoneinfo/zone.tab",
    ];
    CANDIDATES.iter().find_map(|path| {
        let contents = std::fs::read_to_string(Path::new(path)).ok()?;
        find_zone_in_tab(&contents, zone)
    })
}

/// Scan one `.tab` file's contents for `zone` and parse its coordinates.
fn find_zone_in_tab(contents: &str, zone: &str) -> Option<(f64, f64)> {
    contents
        .lines()
        .filter(|l| !l.starts_with('#'))
        .find_map(|line| {
            let mut fields = line.split('\t');
            let _countries = fields.next()?;
            let coords = fields.next()?;
            (fields.next()? == zone)
                .then(|| parse_iso6709(coords))
                .flatten()
        })
}

/// Parse tzdb's ISO 6709 coordinate form: `±DDMM±DDDMM` or
/// `±DDMMSS±DDDMMSS` → (degrees north, degrees east).
fn parse_iso6709(s: &str) -> Option<(f64, f64)> {
    let lon_start = s[1..].find(['+', '-'])? + 1;
    let (lat_s, lon_s) = s.split_at(lon_start);
    Some((
        parse_iso6709_component(lat_s, 2)?,
        parse_iso6709_component(lon_s, 3)?,
    ))
}

fn parse_iso6709_component(s: &str, deg_digits: usize) -> Option<f64> {
    let sign = match s.as_bytes().first()? {
        b'+' => 1.0,
        b'-' => -1.0,
        _ => return None,
    };
    let digits = &s[1..];
    if !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    // DDMM or DDMMSS (one more leading degree digit for longitude).
    if digits.len() != deg_digits + 2 && digits.len() != deg_digits + 4 {
        return None;
    }
    let deg: f64 = digits[..deg_digits].parse().ok()?;
    let min: f64 = digits[deg_digits..deg_digits + 2].parse().ok()?;
    let sec: f64 = if digits.len() == deg_digits + 4 {
        digits[deg_digits + 2..].parse().ok()?
    } else {
        0.0
    };
    Some(sign * (deg + min / 60.0 + sec / 3600.0))
}

/// Compute today's sunrise/sunset for (`lat` °N, `lon` °E) as minutes since
/// local midnight, given the current unix time and the observer's UTC
/// offset. The standard sunrise equation (NOAA/Wikipedia form) — accurate
/// to a couple of minutes, far inside the tz-coordinate error budget.
pub fn solar_events(lat: f64, lon: f64, unix_secs: i64, utc_offset_secs: i32) -> SolarEvents {
    let sin_d = f64::to_radians;
    let sind = |x: f64| sin_d(x).sin();
    let cosd = |x: f64| sin_d(x).cos();

    // Days since J2000 (2000-01-01 12:00 UTC), rounded to the nearest
    // transit; 0.0008 is the fractional-day correction for leap seconds
    // and terrestrial time.
    let jdate = unix_secs as f64 / 86400.0 + 2440587.5;
    let n = (jdate - 2451545.0 + 0.0008).round();

    // Mean solar noon at the observer's meridian (east-positive `lon`
    // pushes solar noon earlier in UTC).
    let j_star = n - lon / 360.0;
    // Solar mean anomaly, equation of center, ecliptic longitude.
    let m = (357.5291 + 0.985_600_28 * j_star).rem_euclid(360.0);
    let c = 1.9148 * sind(m) + 0.0200 * sind(2.0 * m) + 0.0003 * sind(3.0 * m);
    let lambda = (m + c + 180.0 + 102.9372).rem_euclid(360.0);
    // Solar transit (actual local solar noon, as a Julian date).
    let j_transit = 2451545.0 + j_star + 0.0053 * sind(m) - 0.0069 * sind(2.0 * lambda);
    // Declination of the sun.
    let sin_decl = sind(lambda) * sind(23.4397);
    let cos_decl = (1.0 - sin_decl * sin_decl).sqrt();
    // Hour angle for the standard −0.833° zenith correction (refraction +
    // solar disc radius).
    let cos_omega = (sind(-0.833) - sind(lat) * sin_decl) / (cosd(lat) * cos_decl);
    if cos_omega > 1.0 {
        return SolarEvents::PolarNight;
    }
    if cos_omega < -1.0 {
        return SolarEvents::PolarDay;
    }
    let omega = cos_omega.acos().to_degrees();

    let to_local_minutes = |julian: f64| -> f32 {
        let unix = (julian - 2440587.5) * 86400.0;
        let local = unix + f64::from(utc_offset_secs);
        (local.rem_euclid(86400.0) / 60.0) as f32
    };
    SolarEvents::Normal {
        sunrise: to_local_minutes(j_transit - omega / 360.0),
        sunset: to_local_minutes(j_transit + omega / 360.0),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // 2026-06-21T00:00Z and 2026-12-21T00:00Z.
    const JUNE_SOLSTICE: i64 = 1_782_000_000;
    const DECEMBER_SOLSTICE: i64 = 1_797_811_200;

    fn day_length_hours(events: SolarEvents) -> f32 {
        match events {
            SolarEvents::Normal { sunrise, sunset } => (sunset - sunrise).rem_euclid(1440.0) / 60.0,
            _ => panic!("expected a normal day, got {events:?}"),
        }
    }

    #[test]
    fn zone_from_path_handles_macos_and_suffixed_layouts() {
        assert_eq!(
            zone_from_path("/var/db/timezone/zoneinfo/America/Los_Angeles").as_deref(),
            Some("America/Los_Angeles")
        );
        assert_eq!(
            zone_from_path("/usr/share/zoneinfo.default/Europe/Paris").as_deref(),
            Some("Europe/Paris")
        );
        assert_eq!(
            zone_from_path("/usr/share/zoneinfo/America/Argentina/Buenos_Aires").as_deref(),
            Some("America/Argentina/Buenos_Aires")
        );
        assert_eq!(zone_from_path("/usr/share/zoneinfo/"), None);
        assert_eq!(zone_from_path("/etc/whatever"), None);
    }

    #[test]
    fn iso6709_parses_both_precisions() {
        // Los Angeles, from the real zone.tab line.
        let (lat, lon) = parse_iso6709("+340308-1181434").unwrap();
        assert!((lat - 34.052).abs() < 0.01, "lat {lat}");
        assert!((lon + 118.243).abs() < 0.01, "lon {lon}");
        // Sydney (DDMM form), southern hemisphere.
        let (lat, lon) = parse_iso6709("-3352+15113").unwrap();
        assert!((lat + 33.867).abs() < 0.01, "lat {lat}");
        assert!((lon - 151.217).abs() < 0.01, "lon {lon}");
        assert_eq!(parse_iso6709("garbage"), None);
        assert_eq!(parse_iso6709("+34-118x"), None);
    }

    #[test]
    fn tab_lines_resolve_zones() {
        let tab = "# comment\n\
                   US\t+340308-1181434\tAmerica/Los_Angeles\tPacific\n\
                   AU\t-3352+15113\tAustralia/Sydney\tNSW\n";
        let (lat, _) = find_zone_in_tab(tab, "America/Los_Angeles").unwrap();
        assert!(lat > 0.0);
        let (lat, _) = find_zone_in_tab(tab, "Australia/Sydney").unwrap();
        assert!(lat < 0.0, "southern hemisphere must read negative");
        assert_eq!(find_zone_in_tab(tab, "Etc/UTC"), None);
    }

    #[test]
    fn solstice_day_lengths_are_hemisphere_correct() {
        let la = (34.05, -118.24, -7 * 3600); // PDT in June
        let sydney = (-33.87, 151.21, 10 * 3600); // AEST in June

        // June: long days at 34°N (~14.4h), short at 34°S (~9.9h).
        let la_june = day_length_hours(solar_events(la.0, la.1, JUNE_SOLSTICE + 72_000, la.2));
        assert!((14.0..15.0).contains(&la_june), "LA June {la_june}h");
        let syd_june = day_length_hours(solar_events(
            sydney.0,
            sydney.1,
            JUNE_SOLSTICE + 7_200,
            sydney.2,
        ));
        assert!((9.4..10.4).contains(&syd_june), "Sydney June {syd_june}h");

        // December: reversed.
        let la_dec = day_length_hours(solar_events(
            la.0,
            la.1,
            DECEMBER_SOLSTICE + 72_000,
            -8 * 3600, // PST
        ));
        assert!((9.4..10.4).contains(&la_dec), "LA December {la_dec}h");
        let syd_dec = day_length_hours(solar_events(
            sydney.0,
            sydney.1,
            DECEMBER_SOLSTICE + 7_200,
            11 * 3600, // AEDT
        ));
        assert!(
            (14.0..15.0).contains(&syd_dec),
            "Sydney December {syd_dec}h"
        );
    }

    #[test]
    fn la_june_sunrise_lands_in_the_expected_window() {
        // LA 2026-06-21 (PDT): sunrise ≈ 05:41, sunset ≈ 20:08. Assert a
        // ±15-minute window — the equation is good to a couple of minutes;
        // the window guards the plumbing (offsets, signs), not astronomy.
        let SolarEvents::Normal { sunrise, sunset } =
            solar_events(34.05, -118.24, JUNE_SOLSTICE + 72_000, -7 * 3600)
        else {
            panic!("LA is not polar");
        };
        assert!(
            (5.0 * 60.0 + 26.0..5.0 * 60.0 + 56.0).contains(&sunrise),
            "sunrise {sunrise}min"
        );
        assert!(
            (19.0 * 60.0 + 53.0..20.0 * 60.0 + 23.0).contains(&sunset),
            "sunset {sunset}min"
        );
    }

    #[test]
    fn equator_days_are_near_twelve_hours_year_round() {
        for unix in [JUNE_SOLSTICE, DECEMBER_SOLSTICE] {
            let len = day_length_hours(solar_events(0.0, 0.0, unix + 43_200, 0));
            assert!((11.5..12.5).contains(&len), "equator day {len}h");
        }
    }

    #[test]
    fn polar_latitudes_return_polar_day_and_night() {
        // Tromsø, Norway (69.65°N): midnight sun in June, polar night in
        // December.
        assert_eq!(
            solar_events(69.65, 18.96, JUNE_SOLSTICE + 43_200, 2 * 3600),
            SolarEvents::PolarDay
        );
        assert_eq!(
            solar_events(69.65, 18.96, DECEMBER_SOLSTICE + 43_200, 3600),
            SolarEvents::PolarNight
        );
    }

    /// On real machines (macOS/Linux) the OS tzdb should resolve a known
    /// zone; guards the candidate-path list against OS layout drift.
    #[test]
    fn os_tzdb_resolves_a_known_zone() {
        let Some((lat, lon)) = zone_coordinates("America/Los_Angeles") else {
            panic!("OS tzdb tables missing America/Los_Angeles — update CANDIDATES");
        };
        assert!((lat - 34.05).abs() < 0.1);
        assert!((lon + 118.24).abs() < 0.1);
    }
}
