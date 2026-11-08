//! The public calendar issuer keys live on.
//!
//! Every token issued under one issuer key looks like every other, so the key
//! is what decides who a token hides among. If the windows a key issues and is
//! accepted in were the gateway's (or its database's) to choose, an old key
//! could be made current again, a rotation could be timed around one client,
//! or an expired key's acceptance could be stretched. So the windows are not a
//! choice: they are a pure function of a public calendar that the gateway and
//! every client compute from the same constants, here.
//!
//! The calendar is a sequence of **epochs** of [`EPOCH_SECS`] each, the first
//! starting at [`ANCHOR_UNIX_SECS`]. Epoch `n`'s key:
//!
//! - issues from `ANCHOR + n·EPOCH` until `ANCHOR + (n+1)·EPOCH`
//!   (`issue_from` inclusive, `issue_until` exclusive), and
//! - is accepted until `ANCHOR + (n+2)·EPOCH` (exclusive): one further epoch.
//!
//! With the anchor at the Unix epoch, a boundary is any instant whose Unix time
//! is a multiple of 604 800 seconds (a Thursday, 00:00 UTC), and the epoch an
//! instant falls in is `floor(unix_seconds / 604800)`. Anyone can check a
//! key's windows with that one division.
//!
//! An honest gateway has **exactly one key per epoch** and never rotates
//! within one. Nothing in this module enforces that (it holds no state);
//! the gateway's key provisioning and the client's per-epoch record do.

/// The length of one epoch: seven days. A key issues for one epoch and is
/// accepted for one more.
pub const EPOCH_SECS: u64 = 7 * 86_400;

/// The Unix time at which epoch 0 begins: the Unix epoch itself, so that a
/// boundary is any multiple of [`EPOCH_SECS`].
pub const ANCHOR_UNIX_SECS: u64 = 0;

/// The production calendar.
pub const ISSUER_CALENDAR: IssuerCalendar = IssuerCalendar {
    anchor_unix_secs: ANCHOR_UNIX_SECS,
    epoch_secs: EPOCH_SECS,
};

/// A calendar of fixed-length epochs from an anchor. Production uses
/// [`ISSUER_CALENDAR`] and only that; the type exists so a test can stand a
/// boundary a few seconds from now without waiting a week.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IssuerCalendar {
    /// Unix seconds at which epoch 0 begins.
    pub anchor_unix_secs: u64,
    /// Length of each epoch in seconds. Must be non-zero.
    pub epoch_secs: u64,
}

/// One epoch's three instants, in Unix seconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpochWindows {
    /// First instant the key issues (inclusive).
    pub issue_from: u64,
    /// First instant the key no longer issues (exclusive end of issuance).
    pub issue_until: u64,
    /// First instant the key's tokens are no longer accepted.
    pub accept_until: u64,
}

impl IssuerCalendar {
    /// The epoch the instant `unix_secs` falls in, or `None` before the anchor
    /// (or for a calendar with a zero-length epoch).
    pub const fn epoch_at(&self, unix_secs: u64) -> Option<u64> {
        if self.epoch_secs == 0 || unix_secs < self.anchor_unix_secs {
            return None;
        }
        Some((unix_secs - self.anchor_unix_secs) / self.epoch_secs)
    }

    /// Epoch `epoch`'s windows, or `None` if they do not fit in a `u64`.
    pub const fn windows(&self, epoch: u64) -> Option<EpochWindows> {
        let Some(offset) = epoch.checked_mul(self.epoch_secs) else {
            return None;
        };
        let Some(issue_from) = self.anchor_unix_secs.checked_add(offset) else {
            return None;
        };
        let Some(issue_until) = issue_from.checked_add(self.epoch_secs) else {
            return None;
        };
        let Some(accept_until) = issue_until.checked_add(self.epoch_secs) else {
            return None;
        };
        Some(EpochWindows {
            issue_from,
            issue_until,
            accept_until,
        })
    }

    /// The epoch whose windows are exactly `windows`, or `None` if they are
    /// not one epoch's windows on this calendar. This is the whole check: a
    /// key is on the calendar iff this answers `Some`.
    pub const fn epoch_of(&self, windows: EpochWindows) -> Option<u64> {
        let Some(epoch) = self.epoch_at(windows.issue_from) else {
            return None;
        };
        match self.windows(epoch) {
            Some(canonical)
                if canonical.issue_from == windows.issue_from
                    && canonical.issue_until == windows.issue_until
                    && canonical.accept_until == windows.accept_until =>
            {
                Some(epoch)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WEEK: u64 = 604_800;

    #[test]
    fn the_production_calendar_is_seven_day_epochs_from_the_unix_epoch() {
        assert_eq!(ISSUER_CALENDAR.epoch_secs, WEEK);
        assert_eq!(ISSUER_CALENDAR.anchor_unix_secs, 0);
        // 2026-11-05T00:00:00Z, a Thursday, is a boundary.
        let boundary = 1_793_836_800;
        assert_eq!(boundary % WEEK, 0);
        let n = ISSUER_CALENDAR.epoch_at(boundary).unwrap();
        assert_eq!(ISSUER_CALENDAR.epoch_at(boundary - 1), Some(n - 1));
        assert_eq!(ISSUER_CALENDAR.epoch_at(boundary + WEEK - 1), Some(n));
        assert_eq!(
            ISSUER_CALENDAR.windows(n),
            Some(EpochWindows {
                issue_from: boundary,
                issue_until: boundary + WEEK,
                accept_until: boundary + 2 * WEEK,
            })
        );
    }

    #[test]
    fn only_an_epochs_exact_windows_are_on_the_calendar() {
        let cal = ISSUER_CALENDAR;
        let n = 2_966;
        let w = cal.windows(n).unwrap();
        assert_eq!(cal.epoch_of(w), Some(n));

        // Every way a window could be edited is off the calendar: an issuing
        // window moved or stretched, an acceptance window stretched or cut, a
        // rotation shifted by a second.
        let edits = [
            EpochWindows {
                issue_from: w.issue_from + 1,
                ..w
            },
            EpochWindows {
                issue_from: w.issue_from - WEEK,
                ..w
            },
            EpochWindows {
                issue_until: w.issue_until + 1,
                ..w
            },
            EpochWindows {
                issue_until: w.issue_until - 1,
                ..w
            },
            EpochWindows {
                accept_until: w.accept_until + 1,
                ..w
            },
            EpochWindows {
                accept_until: w.accept_until + WEEK,
                ..w
            },
            EpochWindows {
                accept_until: w.issue_until,
                ..w
            },
        ];
        for edited in edits {
            assert_eq!(cal.epoch_of(edited), None, "{edited:?} must be refused");
        }
    }

    #[test]
    fn a_calendar_answers_none_rather_than_wrapping() {
        let cal = ISSUER_CALENDAR;
        assert_eq!(cal.windows(u64::MAX), None);
        let zero = IssuerCalendar {
            anchor_unix_secs: 0,
            epoch_secs: 0,
        };
        assert_eq!(zero.epoch_at(10), None);
        let later = IssuerCalendar {
            anchor_unix_secs: 100,
            epoch_secs: 10,
        };
        assert_eq!(later.epoch_at(99), None);
        assert_eq!(later.epoch_at(100), Some(0));
        assert_eq!(
            later.epoch_of(EpochWindows {
                issue_from: 110,
                issue_until: 120,
                accept_until: 130
            }),
            Some(1)
        );
    }
}
