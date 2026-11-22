//! The runtime seam: spawning, sleeping, timeouts, and the clocks they read.
//!
//! Natively this is tokio and `std::time`, re-exported unchanged, so a native
//! build compiles to exactly what it did before. wasm32 has neither a tokio
//! runtime (its timer driver does not run in a browser) nor a `std` clock (it
//! panics there): futures are driven by the browser's event loop through
//! `wasm_bindgen_futures::spawn_local`, timers are `setTimeout`, and the clocks
//! are `web_time`'s.
//!
//! **A spawned task runs to completion whether or not anyone awaits its
//! handle**, on both targets. The `AppCore` wrappers and the background loops
//! rely on it: work a caller stops waiting for (a turn holding a spend, a
//! regeneration guard) is finished, never cancelled halfway.
//!
//! On wasm32 nothing here requires `Send`, because the browser runs one thread
//! and its futures (`JsFuture`, the fetch-backed reqwest) are not `Send`.

#[cfg(not(target_family = "wasm"))]
mod imp {
    pub use std::time::{SystemTime, UNIX_EPOCH};
    pub use tokio::runtime::Runtime;
    pub use tokio::task::{JoinError, JoinHandle};
    pub use tokio::time::{Instant, sleep, timeout, timeout_at};

    /// The runtime an `AppCore` runs its work on.
    pub fn new_runtime() -> std::io::Result<Runtime> {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_stack_size(8 * 1024 * 1024) // 8 MB — matches default main-thread size
            .build()
    }

    /// Spawn a task on the ambient runtime.
    pub fn spawn<F>(future: F) -> JoinHandle<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        tokio::spawn(future)
    }

    /// Whether there is a runtime to spawn onto: a background chore asked
    /// for from a plain thread (a synchronous unit test) is skipped rather
    /// than panicking.
    pub fn in_runtime() -> bool {
        tokio::runtime::Handle::try_current().is_ok()
    }
}

#[cfg(target_family = "wasm")]
mod imp {
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll};
    use std::time::Duration;

    use futures_util::future::{Either, select};
    use wasm_bindgen::prelude::*;

    pub use web_time::{Instant, SystemTime, UNIX_EPOCH};

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_name = setTimeout)]
        fn set_timeout(handler: &js_sys::Function, timeout_ms: i32) -> JsValue;
    }

    /// The browser's event loop. It needs no handle; this stands where
    /// tokio's `Runtime` does so the `AppCore` wrappers read the same on both
    /// targets.
    pub struct Runtime;

    pub fn new_runtime() -> std::io::Result<Runtime> {
        Ok(Runtime)
    }

    impl Runtime {
        pub fn spawn<F>(&self, future: F) -> JoinHandle<F::Output>
        where
            F: Future + 'static,
            F::Output: 'static,
        {
            spawn(future)
        }
    }

    /// The task ended without its output. A panic aborts on wasm32, so
    /// nothing a caller can observe produces one; it keeps tokio's shape.
    #[derive(Debug)]
    pub struct JoinError;

    impl std::fmt::Display for JoinError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("task ended without a result")
        }
    }

    /// A spawned task's output. Dropping it detaches the task, which runs on.
    pub struct JoinHandle<T>(tokio::sync::oneshot::Receiver<T>);

    impl<T> Future for JoinHandle<T> {
        type Output = Result<T, JoinError>;

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            Pin::new(&mut self.0)
                .poll(cx)
                .map(|r| r.map_err(|_| JoinError))
        }
    }

    /// Spawn a task on the event loop.
    pub fn spawn<F>(future: F) -> JoinHandle<F::Output>
    where
        F: Future + 'static,
        F::Output: 'static,
    {
        let (tx, rx) = tokio::sync::oneshot::channel();
        wasm_bindgen_futures::spawn_local(async move {
            let _ = tx.send(future.await);
        });
        JoinHandle(rx)
    }

    /// The event loop is always there.
    pub fn in_runtime() -> bool {
        true
    }

    /// Wait `duration` on a `setTimeout`, clamped to the longest delay it
    /// takes (about 24.8 days).
    pub async fn sleep(duration: Duration) {
        let ms = i32::try_from(duration.as_millis()).unwrap_or(i32::MAX);
        let fired = js_sys::Promise::new(&mut |resolve, _reject| {
            set_timeout(&resolve, ms);
        });
        let _ = wasm_bindgen_futures::JsFuture::from(fired).await;
    }

    /// The deadline passed first.
    #[derive(Debug)]
    pub struct Elapsed;

    impl std::fmt::Display for Elapsed {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("deadline has elapsed")
        }
    }

    /// `future`'s output, unless `duration` passes first; then `future` is
    /// dropped, as tokio's `timeout` drops it.
    pub async fn timeout<F: Future>(duration: Duration, future: F) -> Result<F::Output, Elapsed> {
        match select(std::pin::pin!(future), std::pin::pin!(sleep(duration))).await {
            Either::Left((output, _)) => Ok(output),
            Either::Right(_) => Err(Elapsed),
        }
    }

    pub async fn timeout_at<F: Future>(deadline: Instant, future: F) -> Result<F::Output, Elapsed> {
        timeout(deadline.saturating_duration_since(Instant::now()), future).await
    }
}

pub(crate) use imp::*;
