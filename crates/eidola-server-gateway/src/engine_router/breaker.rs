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
    /// A probe passed after it was open: one request at a time, until one is
    /// served (healthy) or fails (open again, `backoff` doubled).
    HalfOpen { backoff: Duration },
}

#[derive(Debug, Clone)]
pub struct Breaker {
    state: State,
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
        }
    }

    /// Whether a request may be routed to the upstream, which has
    /// `in_flight` requests running.
    pub fn routable(&self, in_flight: usize) -> bool {
        match self.state {
            State::Healthy { .. } => true,
            State::HalfOpen { .. } => in_flight == 0,
            State::Unverified | State::Open { .. } => false,
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
            State::HalfOpen { backoff } => {
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
            State::HalfOpen { backoff } => self.open(now, (backoff * 2).min(config.max_backoff)),
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
                State::Open { backoff, .. } => State::HalfOpen { backoff },
            };
            return;
        }
        let backoff = match self.state {
            State::Open { backoff, .. } | State::HalfOpen { backoff } => {
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
        assert!(!b.routable(0));
        assert!(b.probe_due(now));
        b.on_probe(true, now, &CONFIG);
        assert!(b.routable(0));

        let mut failed = Breaker::new();
        failed.on_probe(false, now, &CONFIG);
        assert!(!failed.routable(0));
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
        assert!(b.routable(0), "two consecutive failures stay below three");
        b.on_failure(now, &CONFIG);
        assert!(!b.routable(0));
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
        assert!(!b.routable(0));
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
        assert!(b.routable(0));
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
            assert!(b.routable(0), "half-open takes a request");
            assert!(!b.routable(1), "one at a time");
            // A periodic probe passing again changes nothing.
            b.on_probe(true, now, &CONFIG);
            assert!(b.routable(0));
            b.on_failure(now, &CONFIG);
            assert!(!b.routable(0), "one failure opens it again");
            expected = (expected * 2).min(CONFIG.max_backoff);
        }
        assert_eq!(expected, CONFIG.max_backoff);
        now += expected;
        b.on_probe(true, now, &CONFIG);
        b.on_success();
        assert!(b.routable(5), "a served request closes it");
        // Healthy again: back to the threshold and the initial backoff.
        b.on_failure(now, &CONFIG);
        b.on_failure(now, &CONFIG);
        assert!(b.routable(0));
        b.on_failure(now, &CONFIG);
        assert!(!b.routable(0));
        assert!(b.probe_due(now + CONFIG.initial_backoff));
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
        assert!(!b.routable(0));

        let mut c = Breaker::new();
        c.on_probe(true, now, &CONFIG);
        c.on_probe(false, now, &CONFIG);
        assert!(!c.routable(0));
    }
}
