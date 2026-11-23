//! The self-update, on the shell: app-core's updater
//! (`eidola_app_core::updater::mobile`) stages through the contract's
//! `bundle.*` calls.
//!
//! The order of things, with the user in the middle:
//!
//! 1. [`check`]: app-core finds the next release and verifies it exactly as
//!    the desktop does, then reads the bundle it records. Nothing is
//!    downloaded beyond the signed documents, and nothing reaches the shell.
//! 2. The interface shows the release, who attested it, any claims change,
//!    and the digest the shell's sheet will show; the user approves or not.
//! 3. [`install`]: the bundle is downloaded over the shell's pipe (plain
//!    WebPKI TLS, inside this process), its container and then every file
//!    checked against the manifest, and only then `bundle.begin`, a
//!    `bundle.write` per file, `bundle.seal` (whose digest, file count and
//!    length must be the manifest's) and `bundle.activate`.
//! 4. The app reloads (`location.reload()`); the shell boots the new bundle
//!    on trial, and its boot confirms it (`crate::boot`). One that never
//!    confirms is reverted by the shell at the next launch; [`report`] says
//!    where that stands, and [`revert`] is the user's own way back.

use eidola_app_core::AppCore;
use eidola_app_core::error::AppError;
use eidola_app_core::storage::HostFuture;
use eidola_app_core::updater::mobile::{
    ActivatedBundle, BundleReport, BundleState, BundleStore, BundleStoreError, BundleUpdate,
    PendingBundleUpdate, SealedBundle,
};

use crate::client::{ShellClient, Transport};
use crate::protocol::{Code, MAX_REQUEST_BYTES, ShellError};

/// The shell's bundle store as app-core's [`BundleStore`].
pub struct ShellBundles<T> {
    client: ShellClient<T>,
}

impl<T: Transport> ShellBundles<T> {
    pub fn new(client: ShellClient<T>) -> Self {
        Self { client }
    }
}

fn store_error(error: ShellError) -> BundleStoreError {
    match error.code() {
        Some(Code::NotFound) => BundleStoreError::NotFound(error.to_string()),
        Some(Code::InvalidState) => BundleStoreError::InvalidState(error.to_string()),
        _ => BundleStoreError::Failed(error.to_string()),
    }
}

impl<T: Transport> BundleStore for ShellBundles<T> {
    fn begin<'a>(&'a self, version: &'a str) -> HostFuture<'a, Result<(), BundleStoreError>> {
        Box::pin(async move { self.client.bundle_begin(version).await.map_err(store_error) })
    }

    fn write<'a>(
        &'a self,
        version: &'a str,
        path: &'a str,
        offset: u64,
        data: &'a [u8],
    ) -> HostFuture<'a, Result<(), BundleStoreError>> {
        Box::pin(async move {
            if data.is_empty() {
                // Creates the file, as an empty write does.
                return self
                    .client
                    .bundle_write(version, path, offset, &[])
                    .await
                    .map_err(store_error);
            }
            let mut at = offset;
            for chunk in data.chunks(MAX_REQUEST_BYTES) {
                self.client
                    .bundle_write(version, path, at, chunk)
                    .await
                    .map_err(store_error)?;
                at += chunk.len() as u64;
            }
            Ok(())
        })
    }

    fn seal<'a>(
        &'a self,
        version: &'a str,
    ) -> HostFuture<'a, Result<SealedBundle, BundleStoreError>> {
        Box::pin(async move { self.client.bundle_seal(version).await.map_err(store_error) })
    }

    fn activate<'a>(
        &'a self,
        version: &'a str,
    ) -> HostFuture<'a, Result<BundleState, BundleStoreError>> {
        Box::pin(async move {
            self.client
                .bundle_activate(version)
                .await
                .map_err(store_error)
        })
    }

    fn remove<'a>(&'a self, version: &'a str) -> HostFuture<'a, Result<(), BundleStoreError>> {
        Box::pin(async move {
            self.client
                .bundle_remove(version)
                .await
                .map_err(store_error)
        })
    }

    fn state(&self) -> HostFuture<'_, Result<BundleState, BundleStoreError>> {
        Box::pin(async move { self.client.bundle_state().await.map_err(store_error) })
    }
}

/// Find and verify the next release, and read its bundle. Stages nothing.
pub async fn check(core: &AppCore) -> Result<BundleUpdate, AppError> {
    core.prepare_bundle_update().await
}

/// The user approved `pending`: check it byte for byte, stage it in the
/// shell and activate it. The caller reloads next.
pub async fn install<T: Transport>(
    core: &AppCore,
    client: &ShellClient<T>,
    pending: PendingBundleUpdate,
) -> Result<ActivatedBundle, AppError> {
    core.install_bundle_update(pending, &ShellBundles::new(client.clone()))
        .await
}

/// Where the shell's provisional activation stands.
pub async fn report<T: Transport>(client: &ShellClient<T>) -> Result<BundleReport, ShellError> {
    Ok(BundleReport::from_state(&client.bundle_state().await?))
}

/// Boot the embedded bundle from the next load on, confirmed.
pub async fn revert<T: Transport>(client: &ShellClient<T>) -> Result<BundleState, AppError> {
    eidola_app_core::updater::mobile::revert_to_embedded(&ShellBundles::new(client.clone()))
        .await
        .map_err(|e| AppError::Update {
            message: e.to_string(),
        })
}
