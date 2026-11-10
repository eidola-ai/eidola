//! One upstream's health, as this gateway has observed it.
//!
//! Local to the process and never written anywhere: placement data says
//! where an engine is, never whether it is well. A new upstream takes no
//! traffic until a probe has shown it serving the pinned weights; a healthy
//! one opens after consecutive request failures (or at once on a probe
//! failure or a refusal that means it is misconfigured); an open one takes no
//! traffic until its backoff has passed and a probe succeeds, the backoff
//! doubling with every failed probe up to a ceiling.

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

    /// Whether requests may be routed to the upstream.
    pub fn routable(&self) -> bool {
        matches!(self.state, State::Healthy { .. })
    }

    /// Whether a probe should run now: always for a new or healthy upstream
    /// (the periodic check), and for an open one once its backoff has passed.
    pub fn probe_due(&self, now: Instant) -> bool {
        match self.state {
            State::Unverified | State::Healthy { .. } => true,
            State::Open { until, .. } => now >= until,
        }
    }

    /// A request the upstream served.
    pub fn on_success(&mut self) {
        if let State::Healthy { failures } = &mut self.state {
            *failures = 0;
        }
    }

    /// A request the upstream failed. Opens it at the threshold.
    pub fn on_failure(&mut self, now: Instant, config: &BreakerConfig) {
        if let State::Healthy { failures } = &mut self.state {
            *failures += 1;
            if *failures >= config.failure_threshold {
                self.open(now, config.initial_backoff);
            }
        }
    }

    /// A refusal that means the upstream is misconfigured (it does not serve
    /// the pinned weights or model, or does not accept the gateway's token):
    /// open at once, whatever the count.
    pub fn trip(&mut self, now: Instant, config: &BreakerConfig) {
        match self.state {
            State::Open { .. } => {}
            _ => self.open(now, config.initial_backoff),
        }
    }

    /// A probe's result.
    pub fn on_probe(&mut self, ok: bool, now: Instant, config: &BreakerConfig) {
        if ok {
            self.state = match self.state {
                // A periodic probe does not reset the count of failed
                // requests: only a served request does.
                healthy @ State::Healthy { .. } => healthy,
                _ => State::Healthy { failures: 0 },
            };
            return;
        }
        let backoff = match self.state {
            State::Open { backoff, .. } => (backoff * 2).min(config.max_backoff),
            _ => config.initial_backoff,
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
