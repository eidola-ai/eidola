//! Booting the app on its shell, and the provisional-activation handshake.
//!
//! 1. `shell.hello`: a shell without the contract this app needs is refused
//!    ([`ShellError::Contract`]) before anything else is asked of it.
//! 2. The profile is loaded whole from the shell's files
//!    ([`HostProfile::load`]) and [`AppCore::on_host`] opens it, with every
//!    request going out through the shell's pipe ([`ShellDialer`]).
//! 3. [`AppCore::open_database`]: the database opens and its schema is one
//!    this build accepts (on a first launch, it is created).
//! 4. [`AppCore::flush_storage`]: everything that wrote has reached the shell
//!    and is synced.
//! 5. `bundle.confirm`.
//!
//! **Boot health is judged here, and only here**: a bundle is healthy once it
//! runs, speaks to its shell, and has the profile open and durable — the
//! whole path from a tap on the icon to a core that can serve the user. The
//! network is no part of it, so a bundle booted offline (or against a
//! gateway that is down) still confirms; a bundle that cannot open the
//! profile never confirms, and the shell falls back at the next launch. The
//! interface that comes next moves the confirmation after its first frame,
//! so a bundle that cannot draw falls back too.

use std::path::PathBuf;
use std::sync::Arc;

use eidola_app_core::AppCore;
use eidola_app_core::error::AppError;
use eidola_app_core::storage::HostProfile;
use eidola_app_core::transport::HostNetwork;
use rustls::pki_types::CertificateDer;

use crate::client::{ShellClient, Transport};
use crate::files::ShellFiles;
use crate::protocol::{BundleState, Hello, ShellError};
use crate::socket::ShellDialer;

/// The profile's name: the key of its single-writer lock (held in process on
/// wasm32). Its files are the shell's file directory; nothing is read from
/// this path.
pub const PROFILE: &str = "eidola";

/// What a boot is given beyond the shell.
#[derive(Default)]
pub struct BootOptions {
    /// Overrides [`PROFILE`]: natively the lock is a real file lock, so a
    /// native test names a directory of its own.
    pub profile: Option<PathBuf>,
    /// TLS roots trusted beside the built-in ones (a development server's
    /// CA); attestation is unaffected.
    pub extra_tls_roots: Vec<CertificateDer<'static>>,
    /// Overrides [`crate::socket::RECEIVE_MAX`].
    pub receive_max: Option<usize>,
}

/// Why a boot failed, by step.
#[derive(Debug, thiserror::Error)]
pub enum BootError {
    #[error("the shell: {0}")]
    Shell(#[from] ShellError),
    #[error("the profile: {0}")]
    Profile(#[from] AppError),
    #[error("confirming the bundle: {0}")]
    Confirm(ShellError),
}

/// A booted app.
pub struct Booted<T> {
    pub core: AppCore,
    pub hello: Hello,
    pub client: ShellClient<T>,
    /// The bundle state after `bundle.confirm`.
    pub bundle: BundleState,
}

/// Boot on `client`'s shell, and confirm the bundle once healthy.
pub async fn boot<T: Transport>(
    client: ShellClient<T>,
    options: BootOptions,
) -> Result<Booted<T>, BootError> {
    let hello = client.hello().await?;
    hello.check()?;
    let core = open(&client, options).await?;
    core.open_database().await?;
    core.flush_storage().await?;
    let bundle = client.bundle_confirm().await.map_err(BootError::Confirm)?;
    Ok(Booted {
        core,
        hello,
        client,
        bundle,
    })
}

/// Load the profile from the shell's files and open a core on it, its
/// network the shell's pipe.
pub async fn open<T: Transport>(
    client: &ShellClient<T>,
    options: BootOptions,
) -> Result<AppCore, AppError> {
    let files = Arc::new(ShellFiles::new(client.clone()));
    let name = options.profile.unwrap_or_else(|| PathBuf::from(PROFILE));
    let profile = HostProfile::load(name, files).await?;
    let mut dialer = ShellDialer::new(client.clone());
    if let Some(max) = options.receive_max {
        dialer = dialer.receive_max(max);
    }
    AppCore::on_host(
        profile,
        Some(HostNetwork {
            dialer: Arc::new(dialer),
            extra_tls_roots: options.extra_tls_roots,
        }),
    )
}
