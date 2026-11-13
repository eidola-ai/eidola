//! The drain barrier every backed test binds, shared by the test crates that
//! build stores over a real `AppCore` (`tests/stores.rs`, `tests/behavior.rs`).

use std::sync::Arc;

use eidola_app_core::AppCore;

/// The keepalive every backed test binds, and the **drain barrier** at its
/// teardown.
///
/// A store's bridge task runs on the `AppCore`'s own tokio runtime holding an
/// `Arc<AppCore>`, and the store entities keep the core alive only until `cx`
/// tears down on this thread. If a bridge task is still running then it becomes
/// the *last* owner and drops the tokio runtime from within its own worker →
/// "Cannot drop a runtime in a context where blocking is not allowed". Waiting
/// for the runtime to go idle first guarantees the last `Arc` always drops on
/// the test thread — the same barrier
/// `record_refresh_supersedes_in_flight_fetch` (`tests/behavior.rs`) takes
/// inline, here as a guard so every test that binds one gets it without a line
/// of its own, on the unwinding path as on the passing one.
///
/// **Bounded, unlike the record test's.** These tasks include network-connect
/// paths, so "the count reaches zero" is not as obviously bounded as a local-DB
/// read; the unreachable base URL fails fast, and a wait that outlives the
/// deadline is reported rather than hung on. The report is skipped while the
/// thread is already panicking — a panic in `Drop` during unwinding aborts the
/// process and would bury the assertion that actually failed.
///
/// **And giving up on the wait leaks the core rather than releasing it.**
/// Timing out is precisely the case where a task may still own an
/// `Arc<AppCore>`, so releasing this one hands that task the last reference and
/// the runtime gets dropped from a worker — the exact panic the barrier exists
/// to prevent, arriving on the path where it does the most damage. See `Drop`.
///
/// **Some tasks never drain, and the barrier has to be told.** The hazard is
/// only ever a task that *owns* an `Arc<AppCore>`; a task that owns none can
/// never be the last owner, so waiting on it buys nothing and would hang out
/// the whole timeout. [`Backing::excuse_ownerless_tasks`] is how a test that
/// spawns one says so, and it is a claim about the task's captures — check them
/// before calling it.
///
/// **Whatever a test holds that keeps a task running is declared after its
/// guard**, so that on every exit — a failed assertion included — it is
/// released before the guard waits: locals drop in reverse order of
/// declaration.
pub struct Backing {
    pub core: Arc<AppCore>,
    pub _dir: tempfile::TempDir,
    /// How many alive tasks are known to hold no `Arc<AppCore>` — see
    /// [`Backing::excuse_ownerless_tasks`].
    pub ownerless: usize,
    /// How long the barrier waits. A field rather than the bare constant so the
    /// guard's own timeout behaviour is reachable from a test in under half a
    /// minute — the only way this class is testable at all, since the hazard it
    /// prevents is a race nothing can schedule.
    pub drain_timeout: std::time::Duration,
}

/// How long teardown waits for the bridge tasks to drain. Generous: the tasks
/// fail fast against an unreachable URL, so reaching this means something is
/// genuinely stuck.
pub const DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

// Each test crate that includes this module uses a different subset of it.
#[allow(dead_code)]
impl Backing {
    /// The guard over `core` with the standard deadline, keeping `dir` (the
    /// core's profile) alive until the runtime has drained.
    pub fn new(core: Arc<AppCore>, dir: tempfile::TempDir) -> Self {
        Self {
            core,
            _dir: dir,
            ownerless: 0,
            drain_timeout: DRAIN_TIMEOUT,
        }
    }

    /// Excuse `n` tasks from the drain barrier, because they hold no
    /// `Arc<AppCore>` and therefore cannot drop the runtime from a worker.
    ///
    /// The live case is `stores::install_bus_bridge`'s tokio half: it captures
    /// a `broadcast::Receiver` and an mpsc `Sender` and nothing else, and it
    /// ends only when its gpui counterpart does — at `cx` teardown, which is
    /// *after* this guard runs. Waiting for it is waiting for something that
    /// cannot happen.
    pub fn excuse_ownerless_tasks(&mut self, n: usize) {
        self.ownerless += n;
    }
}

impl Drop for Backing {
    fn drop(&mut self) {
        let deadline = std::time::Instant::now() + self.drain_timeout;
        while self.core.runtime().metrics().num_alive_tasks() > self.ownerless {
            if std::time::Instant::now() >= deadline {
                // **The invariant, restored the only way still available: if
                // this guard could not prove the runtime idle, nothing may
                // ever drop it.**
                //
                // Giving up on the wait used to hand the hazard straight back
                // — this guard's `Arc` drops, `Stores` drops at `cx` teardown,
                // and the still-running task becomes the last owner and drops
                // the runtime from inside its own worker. On the unwinding
                // path that is a second panic during an unwind, which aborts
                // the process and takes the assertion failure that actually
                // matters with it (Codex review, PR #327).
                //
                // So leak one owner. The strong count can no longer reach
                // zero, the runtime is never dropped, and the class of panic
                // this guard exists to prevent is unreachable rather than
                // merely unlikely. The cost is one abandoned runtime and its
                // threads for the rest of the process — paid only on a path
                // where a test has already failed or is about to, and the
                // alternative is losing the reason it failed.
                std::mem::forget(self.core.clone());
                if !std::thread::panicking() {
                    panic!(
                        "bridge tasks did not drain within {:?} — the runtime has \
                         been leaked so the wait's failure is what you see, rather than a \
                         worker dropping it out from under this thread",
                        self.drain_timeout
                    );
                }
                return;
            }
            std::thread::yield_now();
        }
    }
}
