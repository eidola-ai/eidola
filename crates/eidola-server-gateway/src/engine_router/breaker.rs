//! One upstream's health, as this gateway has observed it.
//!
//! Local to the process and never written anywhere: placement data says
//! where an engine is, never whether it is well. A new upstream takes no
//! traffic until a probe has shown it serving the pinned weights; a healthy
//! one opens after consecutive request failures (or at once on a probe
//! failure or a refusal that means it is misconfigured); an open one takes no
//! traffic until its backoff has passed and a probe succeeds, the backoff
//! doubling with every failed probe up to a ceiling.
//!
//! **A passing probe does not close an open upstream; a served request
//! does.** A probe shows the engine answers `/healthz` and reports the pinned
//! weights, not that it serves chat: one that passes its probes and fails its
//! requests would otherwise open and close on every backoff, at the initial
//! backoff, forever, each time failing requests the gateway returns rather
//! than retries. So a probe that passes moves an open upstream to half-open,
//! which takes one request at a time and keeps the backoff: the first request
//! it serves closes it, and the first it fails opens it again with the
//! backoff doubled.
//!
//! **Admission is decided here, under the breaker's lock, at each attempt.**
//! [`Breaker::admit`] answers whether this request may go to the upstream now
//! and, for a half-open one, takes its single slot in the same step, so two
//! requests cannot both see the slot free; the slot is given back
//! ([`Breaker::release`]) when its request ends, whichever way. A slot taken
//! in one half-open spell is never given back to a later one.

use std::time::{Duration, Instant};

/// The breaker's tuning.
#[derive(Debug, Clone, Copy)]
pub struct BreakerConfig {
    /// Consecutive request failures that open a healthy upstream.
    pub failure_threshold: u32,
    /// The first backoff after opening.
    pub initial_backoff: Duration,
    /// The longest backoff.
    pub max_backoff: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Never probed successfully since it appeared in placement.
    Unverified,
    /// Taking traffic; `failures` consecutive request failures so far.
    Healthy { failures: u32 },
    /// Taking no traffic until `until`, then probed.
    Open { until: Instant, backoff: Duration },
    /// A probe passed after it was open: one request at a time (`taken`),
    /// until one is served (healthy) or fails (open again, `backoff`
    /// doubled). `spell` tells this half-open period from later ones.
    HalfOpen {
        backoff: Duration,
        spell: u64,
        taken: bool,
    },
}

/// What [`Breaker::admit`] let through: handed back to [`Breaker::release`]
/// when the request ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admitted {
    /// A healthy upstream's request.
    Healthy,
    /// A half-open upstream's single request, in the spell it was taken in.
    Trial { spell: u64 },
}

#[derive(Debug, Clone)]
pub struct Breaker {
    state: State,
    /// Half-open spells begun so far.
    spells: u64,
}

impl Default for Breaker {
    fn default() -> Self {
        Self::new()
    }
}

impl Breaker {
    /// A breaker for an upstream that has not been probed yet.
    pub fn new() -> Self {
        Self {
            state: State::Unverified,
            spells: 0,
        }
    }

    /// Whether a request could be admitted now: what ranking filters on. It
    /// is a hint, not the gate; [`Breaker::admit`] is.
    pub fn routable(&self) -> bool {
        match self.state {
            State::Healthy { .. } => true,
            State::HalfOpen { taken, .. } => !taken,
            State::Unverified | State::Open { .. } => false,
        }
    }

    /// Admit one request now, or refuse it: every request to a healthy
    /// upstream, and to a half-open one only while its slot is free, which
    /// this takes.
    pub fn admit(&mut self) -> Option<Admitted> {
        match &mut self.state {
            State::Healthy { .. } => Some(Admitted::Healthy),
            State::HalfOpen { spell, taken, .. } if !*taken => {
                *taken = true;
                Some(Admitted::Trial { spell: *spell })
            }
            _ => None,
        }
    }

    /// An admitted request has ended. A half-open slot taken in the current
    /// spell is free again; nothing else changes (its verdict, if it had one,
    /// was already recorded).
    pub fn release(&mut self, admitted: Admitted) {
        if let (Admitted::Trial { spell: ended }, State::HalfOpen { spell, taken, .. }) =
            (admitted, &mut self.state)
            && ended == *spell
        {
            *taken = false;
        }
    }

    /// Whether a probe should run now: always for a new or healthy upstream
    /// (the periodic check), and for an open one once its backoff has passed.
    pub fn probe_due(&self, now: Instant) -> bool {
        match self.state {
            State::Unverified | State::Healthy { .. } | State::HalfOpen { .. } => true,
            State::Open { until, .. } => now >= until,
        }
    }

    /// A request the upstream served: healthy, with no failures counted.
    pub fn on_success(&mut self) {
        if let State::Healthy { .. } | State::HalfOpen { .. } = self.state {
            self.state = State::Healthy { failures: 0 };
        }
    }

    /// A request the upstream failed. Opens a healthy one at the threshold,
    /// and a half-open one at once, its backoff doubled.
    pub fn on_failure(&mut self, now: Instant, config: &BreakerConfig) {
        match &mut self.state {
            State::Healthy { failures } => {
                *failures += 1;
                if *failures >= config.failure_threshold {
                    self.open(now, config.initial_backoff);
                }
            }
            State::HalfOpen { backoff, .. } => {
                let backoff = (*backoff * 2).min(config.max_backoff);
                self.open(now, backoff);
            }
            State::Unverified | State::Open { .. } => {}
        }
    }

    /// A refusal that means the upstream is misconfigured (it does not serve
    /// the pinned weights or model, or does not accept the gateway's token):
    /// open at once, whatever the count.
    pub fn trip(&mut self, now: Instant, config: &BreakerConfig) {
        match self.state {
            State::Open { .. } => {}
            State::HalfOpen { backoff, .. } => {
                self.open(now, (backoff * 2).min(config.max_backoff))
            }
            State::Unverified | State::Healthy { .. } => self.open(now, config.initial_backoff),
        }
    }

    /// A probe's result.
    pub fn on_probe(&mut self, ok: bool, now: Instant, config: &BreakerConfig) {
        if ok {
            self.state = match self.state {
                State::Unverified => State::Healthy { failures: 0 },
                // A periodic probe does not reset the count of failed
                // requests, nor the backoff: only a served request does.
                kept @ (State::Healthy { .. } | State::HalfOpen { .. }) => kept,
                State::Open { backoff, .. } => {
                    self.spells += 1;
                    State::HalfOpen {
                        backoff,
                        spell: self.spells,
                        taken: false,
                    }
                }
            };
            return;
        }
        let backoff = match self.state {
            State::Open { backoff, .. } | State::HalfOpen { backoff, .. } => {
                (backoff * 2).min(config.max_backoff)
            }
            State::Unverified | State::Healthy { .. } => config.initial_backoff,
        };
        self.open(now, backoff);
    }

    fn open(&mut self, now: Instant, backoff: Duration) {
        self.state = State::Open {
            until: now + backoff,
            backoff,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: BreakerConfig = BreakerConfig {
        failure_threshold: 3,
        initial_backoff: Duration::from_secs(5),
        max_backoff: Duration::from_secs(40),
    };

    #[test]
    fn a_new_upstream_takes_no_traffic_until_a_probe_passes() {
        let now = Instant::now();
        let mut b = Breaker::new();
        assert!(!b.routable());
        assert!(b.probe_due(now));
        b.on_probe(true, now, &CONFIG);
        assert!(b.routable());

        let mut failed = Breaker::new();
        failed.on_probe(false, now, &CONFIG);
        assert!(!failed.routable());
        assert!(!failed.probe_due(now));
        assert!(failed.probe_due(now + CONFIG.initial_backoff));
    }

    #[test]
    fn consecutive_failures_open_and_a_success_resets_the_count() {
        let now = Instant::now();
        let mut b = Breaker::new();
        b.on_probe(true, now, &CONFIG);
        b.on_failure(now, &CONFIG);
        b.on_failure(now, &CONFIG);
        b.on_success();
        b.on_failure(now, &CONFIG);
        b.on_failure(now, &CONFIG);
        assert!(b.routable(), "two consecutive failures stay below three");
        b.on_failure(now, &CONFIG);
        assert!(!b.routable());
        assert!(!b.probe_due(now + CONFIG.initial_backoff / 2));
        assert!(b.probe_due(now + CONFIG.initial_backoff));
    }

    #[test]
    fn a_passing_periodic_probe_keeps_the_failure_count() {
        let now = Instant::now();
        let mut b = Breaker::new();
        b.on_probe(true, now, &CONFIG);
        b.on_failure(now, &CONFIG);
        b.on_failure(now, &CONFIG);
        b.on_probe(true, now, &CONFIG);
        b.on_failure(now, &CONFIG);
        assert!(!b.routable());
    }

    #[test]
    fn failed_probes_double_the_backoff_up_to_the_ceiling() {
        let mut now = Instant::now();
        let mut b = Breaker::new();
        b.on_probe(true, now, &CONFIG);
        b.trip(now, &CONFIG);
        let mut expected = CONFIG.initial_backoff;
        for _ in 0..6 {
            assert!(!b.probe_due(now + expected - Duration::from_millis(1)));
            now += expected;
            assert!(b.probe_due(now));
            b.on_probe(false, now, &CONFIG);
            expected = (expected * 2).min(CONFIG.max_backoff);
        }
        assert_eq!(expected, CONFIG.max_backoff);
        now += expected;
        b.on_probe(true, now, &CONFIG);
        assert!(b.routable());
    }

    /// An upstream that passes its probes and fails its requests: each
    /// passing probe lets one request through at a time and keeps the
    /// backoff, each failed request doubles it, and only a served request
    /// closes the breaker and forgets it.
    #[test]
    fn a_passing_probe_does_not_reset_the_backoff_until_a_request_is_served() {
        let mut now = Instant::now();
        let mut b = Breaker::new();
        b.on_probe(true, now, &CONFIG);
        b.trip(now, &CONFIG);
        let mut expected = CONFIG.initial_backoff;
        for _ in 0..6 {
            assert!(!b.probe_due(now + expected - Duration::from_millis(1)));
            now += expected;
            assert!(b.probe_due(now));
            b.on_probe(true, now, &CONFIG);
            let trial = b.admit().expect("half-open takes a request");
            assert_eq!(b.admit(), None, "one at a time");
            // A periodic probe passing again changes nothing.
            b.on_probe(true, now, &CONFIG);
            assert_eq!(b.admit(), None);
            b.on_failure(now, &CONFIG);
            b.release(trial);
            assert!(!b.routable(), "one failure opens it again");
            assert_eq!(b.admit(), None);
            expected = (expected * 2).min(CONFIG.max_backoff);
        }
        assert_eq!(expected, CONFIG.max_backoff);
        now += expected;
        b.on_probe(true, now, &CONFIG);
        let trial = b.admit().unwrap();
        b.on_success();
        b.release(trial);
        assert!(
            (0..5).all(|_| b.admit() == Some(Admitted::Healthy)),
            "a served request closes it"
        );
        // Healthy again: back to the threshold and the initial backoff.
        b.on_failure(now, &CONFIG);
        b.on_failure(now, &CONFIG);
        assert!(b.routable());
        b.on_failure(now, &CONFIG);
        assert!(!b.routable());
        assert!(b.probe_due(now + CONFIG.initial_backoff));
    }

    /// A half-open slot is taken by admission itself and given back when its
    /// request ends without a verdict; a slot taken in an earlier spell is
    /// never given back to a later one.
    #[test]
    fn a_half_open_slot_is_taken_at_admission_and_given_back_once() {
        let now = Instant::now();
        let later = now + CONFIG.initial_backoff;
        let mut b = Breaker::new();
        b.on_probe(true, now, &CONFIG);
        assert_eq!(b.admit(), Some(Admitted::Healthy));
        b.trip(now, &CONFIG);
        assert_eq!(b.admit(), None, "open admits nothing");
        b.on_probe(true, later, &CONFIG);
        let first = b.admit().unwrap();
        assert!(!b.routable());
        assert_eq!(b.admit(), None);
        // Ended with no verdict (its client left): the slot is free again.
        b.release(first);
        let second = b.admit().unwrap();
        // It fails; a new spell begins after the backoff, and the first
        // spell's slot given back now frees nothing in it.
        b.on_failure(later, &CONFIG);
        let again = later + CONFIG.initial_backoff * 2;
        b.on_probe(true, again, &CONFIG);
        let third = b.admit().unwrap();
        b.release(second);
        assert_eq!(b.admit(), None, "the stale release freed nothing");
        b.release(third);
        assert!(b.admit().is_some());
    }

    /// A failing probe of a half-open upstream doubles the backoff too.
    #[test]
    fn a_failing_probe_of_a_half_open_upstream_doubles_the_backoff() {
        let now = Instant::now();
        let mut b = Breaker::new();
        b.on_probe(true, now, &CONFIG);
        b.trip(now, &CONFIG);
        let later = now + CONFIG.initial_backoff;
        b.on_probe(true, later, &CONFIG);
        b.on_probe(false, later, &CONFIG);
        assert!(!b.probe_due(later + CONFIG.initial_backoff * 2 - Duration::from_millis(1)));
        assert!(b.probe_due(later + CONFIG.initial_backoff * 2));
    }

    #[test]
    fn a_trip_opens_at_once_and_a_failing_healthy_probe_too() {
        let now = Instant::now();
        let mut b = Breaker::new();
        b.on_probe(true, now, &CONFIG);
        b.trip(now, &CONFIG);
        assert!(!b.routable());

        let mut c = Breaker::new();
        c.on_probe(true, now, &CONFIG);
        c.on_probe(false, now, &CONFIG);
        assert!(!c.routable());
    }
}
