//! The durability barrier: nothing leaves the device ahead of what it
//! depends on being on disk.
//!
//! Every request this crate sends may rest on a commit made just before it: a
//! credential marked spending before its spend is sent, a pending issuance
//! before it is asked for, a prefix-cache key's claim before the body carrying
//! it is written. Recovery after a crash reads those rows. If a request could
//! reach the server while the commit it rests on could still be lost, a
//! restart would find no trace of a spend the server has seen, or of an
//! issuance it has answered.
//!
//! [`Durability::barrier`] is the guarantee recovery needs. **Contract: once
//! it returns `Ok`, every database commit the core made before the call is
//! durable on the host.** Each core's transport carries its core's
//! `Durability`, and awaits it before every request leaves the device, on
//! every transport (`transport::RequestBuilder::send`,
//! `transport::Client::execute`), and again after a deferred body has been
//! produced and before its bytes are written (`transport::RequestBuilder::body`).
//!
//! Where the database writes through to the filesystem before a commit
//! returns (the desktop), every commit is already durable when it returns,
//! and the barrier returns at once. A profile its host owns
//! ([`crate::storage`]) commits in memory and streams the writes behind; the
//! barrier is that profile's flush, and no other profile's.
//!
//! **It never waits forever.** It fails, and the request is refused unsent,
//! when the stream has stopped (a host call failed, or the stream's task is
//! gone), and when the host has not made the commits durable within
//! [`BARRIER_TIMEOUT`]: a host that stalls costs a request, not the app. The
//! stream itself is a task of its own that waits on the host alone, never on
//! anything a barrier's caller holds, so the caller's progress cannot be what
//! it waits for.

use std::time::Duration;

use crate::error::AppError;
use crate::storage::ProfileImage;

/// The longest a request waits for the host to make earlier commits durable.
pub(crate) const BARRIER_TIMEOUT: Duration = Duration::from_secs(30);

/// The barrier of one core's profile.
#[derive(Clone, Default)]
pub(crate) struct Durability {
    /// The profile's image when its host owns it; `None` when its files are
    /// written through to the filesystem.
    image: Option<ProfileImage>,
}

impl Durability {
    pub(crate) fn host(image: ProfileImage) -> Self {
        Self { image: Some(image) }
    }

    /// Wait until every database commit made before this call is durable;
    /// fail if it is not, or cannot be, within [`BARRIER_TIMEOUT`].
    pub(crate) async fn barrier(&self) -> Result<(), AppError> {
        self.barrier_within(BARRIER_TIMEOUT).await
    }

    async fn barrier_within(&self, limit: Duration) -> Result<(), AppError> {
        let Some(image) = &self.image else {
            return Ok(());
        };
        match crate::rt::timeout(limit, image.flush()).await {
            Ok(flushed) => flushed,
            Err(_) => Err(AppError::Database {
                message: format!(
                    "the device did not save the profile within {} ms",
                    limit.as_millis()
                ),
            }),
        }
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use std::io;
    use std::sync::Arc;

    use super::*;
    use crate::storage::{HostFuture, HostProfile, HostStorage};

    /// A host that holds every write forever, and has no files.
    struct StalledHost;

    impl HostStorage for StalledHost {
        fn read(&self, _: &str) -> HostFuture<'_, io::Result<Option<Vec<u8>>>> {
            Box::pin(async { Ok(None) })
        }
        fn write_at(&self, _: &str, _: u64, _: Vec<u8>) -> HostFuture<'_, io::Result<()>> {
            Box::pin(std::future::pending())
        }
        fn truncate(&self, _: &str, _: u64) -> HostFuture<'_, io::Result<()>> {
            Box::pin(std::future::pending())
        }
        fn sync(&self, _: &str) -> HostFuture<'_, io::Result<()>> {
            Box::pin(std::future::pending())
        }
        fn rename(&self, _: &str, _: &str) -> HostFuture<'_, io::Result<()>> {
            Box::pin(std::future::pending())
        }
        fn remove(&self, _: &str) -> HostFuture<'_, io::Result<()>> {
            Box::pin(std::future::pending())
        }
        fn list(&self) -> HostFuture<'_, io::Result<Vec<String>>> {
            Box::pin(async { Ok(Vec::new()) })
        }
    }

    /// Every test here must end well inside this, or it has hung.
    const HUNG: Duration = Duration::from_secs(10);

    /// **A host that never answers costs the request, not the app.** The
    /// stream runs but the host holds its write: the barrier fails at its
    /// bound instead of waiting forever.
    #[tokio::test]
    async fn a_stalled_host_fails_the_barrier_at_its_bound() {
        let dir = tempfile::tempdir().unwrap();
        let profile = HostProfile::load(dir.path().to_path_buf(), Arc::new(StalledHost))
            .await
            .unwrap();
        let (image, stream) = profile.into_parts();
        let stream = tokio::spawn(stream);
        image.replace("config.toml", b"x = 1".to_vec()).unwrap();
        let durability = Durability::host(image);
        let barrier = durability.barrier_within(Duration::from_millis(200));
        let outcome = tokio::time::timeout(HUNG, barrier)
            .await
            .expect("the barrier hung");
        assert!(
            outcome.is_err(),
            "nothing reached the host, yet the barrier passed"
        );
        stream.abort();
    }

    /// **A stream whose task is gone stops, even if it never ran.** Its
    /// flush, and so the barrier, fails at once rather than waiting on a task
    /// that will never drain it.
    #[tokio::test]
    async fn a_stream_dropped_before_it_ran_fails_the_barrier_at_once() {
        let dir = tempfile::tempdir().unwrap();
        let profile = HostProfile::load(dir.path().to_path_buf(), Arc::new(StalledHost))
            .await
            .unwrap();
        let (image, stream) = profile.into_parts();
        drop(stream);
        image.replace("config.toml", b"x = 1".to_vec()).unwrap_err();
        let durability = Durability::host(image);
        let barrier = durability.barrier_within(HUNG * 2);
        let outcome = tokio::time::timeout(HUNG, barrier)
            .await
            .expect("the barrier hung");
        assert!(outcome.is_err());
    }

    /// The desktop's barrier: writes through, nothing to wait for.
    #[tokio::test]
    async fn a_profile_on_the_filesystem_passes_at_once() {
        tokio::time::timeout(HUNG, Durability::default().barrier())
            .await
            .expect("the barrier hung")
            .unwrap();
    }
}
