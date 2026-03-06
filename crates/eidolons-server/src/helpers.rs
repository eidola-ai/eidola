//! Consolidated utility functions: calendar and epoch computation.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::error::ServerError;

// ---------------------------------------------------------------------------
// Calendar (Hinnant civil algorithm)
// ---------------------------------------------------------------------------

/// Decompose a unix timestamp (seconds) into (year, month, day).
pub fn civil_from_unix(secs: i64) -> (i64, u64, u64) {
    let days_since_epoch = secs.div_euclid(86_400);
    let z = days_since_epoch + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Convert (year, month, day) to days since 1970-01-01 (inverse Hinnant).
pub fn days_from_civil(y: i64, m: u64, d: u64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400) as u64;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe as i64 - 719468
}

// ---------------------------------------------------------------------------
// Timestamp formatting
// ---------------------------------------------------------------------------

/// Convert a `SystemTime` to signed unix seconds, rounded down to whole seconds.
pub fn system_time_to_unix_seconds(t: SystemTime) -> Option<i64> {
    match t.duration_since(UNIX_EPOCH) {
        Ok(d) => i64::try_from(d.as_secs()).ok(),
        Err(e) => {
            let d = e.duration();
            let secs = i64::try_from(d.as_secs()).ok()?;
            if d.subsec_nanos() == 0 {
                secs.checked_neg()
            } else {
                secs.checked_add(1)?.checked_neg()
            }
        }
    }
}

/// Format signed unix seconds as an ISO 8601 string.
pub fn unix_to_iso(secs: i64) -> String {
    let days_since_epoch = secs.div_euclid(86_400);
    let time_of_day = secs.rem_euclid(86_400) as u64;
    let (hour, min, sec) = (
        time_of_day / 3600,
        (time_of_day % 3600) / 60,
        time_of_day % 60,
    );

    let z = days_since_epoch + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{min:02}:{sec:02}Z")
}

/// Format a `SystemTime` as an ISO 8601 string (e.g. "2026-03-02T12:34:56Z").
pub fn system_time_to_iso(t: SystemTime) -> Result<String, ServerError> {
    let secs = system_time_to_unix_seconds(t).ok_or_else(|| {
        ServerError::Internal("timestamp out of supported range for i64 unix seconds".to_string())
    })?;
    Ok(unix_to_iso(secs))
}

/// Like `system_time_to_iso` but returns an empty string on error.
pub fn system_time_to_iso_lossy(t: SystemTime) -> String {
    system_time_to_iso(t).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Key epoch configuration
// ---------------------------------------------------------------------------

/// Default key epoch duration: 7 days.
pub const DEFAULT_EPOCH_DURATION: Duration = Duration::from_secs(7 * 86400);

/// Default grace period: credentials remain accepted for this long after
/// the issuance window closes. Defaults to one epoch duration.
pub const DEFAULT_GRACE_PERIOD: Duration = Duration::from_secs(7 * 86400);

/// Configuration for key epoch timing.
#[derive(Debug, Clone)]
pub struct EpochConfig {
    /// How long each key is used for issuance.
    pub epoch_duration: Duration,
    /// How long after `issue_until` the key's credentials remain accepted.
    pub grace_period: Duration,
}

impl Default for EpochConfig {
    fn default() -> Self {
        Self {
            epoch_duration: DEFAULT_EPOCH_DURATION,
            grace_period: DEFAULT_GRACE_PERIOD,
        }
    }
}

impl EpochConfig {
    /// Compute `issue_until` and `accept_until` given an `issue_from` timestamp.
    pub fn boundaries_from(&self, issue_from: SystemTime) -> (SystemTime, SystemTime) {
        let issue_until = issue_from + self.epoch_duration;
        let accept_until = issue_until + self.grace_period;
        (issue_until, accept_until)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unix_to_iso_handles_epoch_and_pre_epoch() {
        assert_eq!(unix_to_iso(0), "1970-01-01T00:00:00Z");
        assert_eq!(unix_to_iso(-1), "1969-12-31T23:59:59Z");
        assert_eq!(unix_to_iso(86_400), "1970-01-02T00:00:00Z");
    }

    #[test]
    fn system_time_to_iso_handles_pre_epoch() {
        let t = UNIX_EPOCH
            .checked_sub(Duration::from_secs(1))
            .expect("valid pre-epoch timestamp");
        assert_eq!(
            system_time_to_iso(t).expect("timestamp should format"),
            "1969-12-31T23:59:59Z"
        );
    }

    #[test]
    fn system_time_to_iso_floors_subsecond_pre_epoch() {
        let t = UNIX_EPOCH
            .checked_sub(Duration::from_millis(1))
            .expect("valid pre-epoch timestamp");
        assert_eq!(
            system_time_to_iso(t).expect("timestamp should format"),
            "1969-12-31T23:59:59Z"
        );
    }

    #[test]
    fn test_epoch_config_boundaries() {
        let config = EpochConfig::default();
        let issue_from = UNIX_EPOCH + Duration::from_secs(days_from_civil(2026, 3, 1) as u64 * 86400);
        let (issue_until, accept_until) = config.boundaries_from(issue_from);
        assert_eq!(system_time_to_iso_lossy(issue_from), "2026-03-01T00:00:00Z");
        assert_eq!(system_time_to_iso_lossy(issue_until), "2026-03-08T00:00:00Z");
        assert_eq!(system_time_to_iso_lossy(accept_until), "2026-03-15T00:00:00Z");
    }

    #[test]
    fn test_epoch_config_custom_durations() {
        let config = EpochConfig {
            epoch_duration: Duration::from_secs(30 * 86400),
            grace_period: Duration::from_secs(3 * 86400),
        };
        let issue_from = UNIX_EPOCH + Duration::from_secs(days_from_civil(2026, 3, 1) as u64 * 86400);
        let (issue_until, accept_until) = config.boundaries_from(issue_from);
        assert_eq!(system_time_to_iso_lossy(issue_until), "2026-03-31T00:00:00Z");
        assert_eq!(system_time_to_iso_lossy(accept_until), "2026-04-03T00:00:00Z");
    }

    #[test]
    fn test_civil_roundtrip() {
        let days = days_from_civil(2026, 3, 4);
        let secs = days * 86400;
        let (y, m, d) = civil_from_unix(secs);
        assert_eq!((y, m, d), (2026, 3, 4));
    }

}
