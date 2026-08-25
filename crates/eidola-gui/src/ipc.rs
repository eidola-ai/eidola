//! The local control socket — the door another process knocks on.
//!
//! The local database has exactly one writer, and this process is holding it
//! (`AppError::DatabaseInUse` is what a second opener gets). That is the whole
//! reason this socket exists: with the app running, a command-line invocation
//! cannot open the profile itself, so it asks the process that already has it.
//!
//! **The flock holder owns the socket.** This module runs in the process that
//! took the lock, so a socket file already sitting in the data directory is
//! stale by construction — nothing else can be listening on it while we hold
//! the lock — and binding replaces it. That inference is the reason the bind
//! does not have to probe the old socket first, and it stops being valid the
//! moment anything binds this path without holding the lock. Do not.
//!
//! ## Where it lives, and who may talk to it
//!
//! [`eidola_app_core::ipc::socket_path`] — beside the database it speaks for,
//! and named there rather than here so whoever dials finds the same file. The data
//! directory is already `0700`, which is the same-user authentication model
//! ssh-agent, gpg-agent and the session bus all use: another user cannot
//! traverse into the directory, so they cannot reach the socket at all. The
//! socket is then `0600` as well — belt and braces on a path where the belt is
//! doing the work.
//!
//! On top of that, **tier 0 peer authentication**: every accepted connection's
//! credentials are read from the kernel (`SO_PEERCRED` / `LOCAL_PEERCRED`, via
//! [`tokio::net::UnixStream::peer_cred`]) and the peer's uid must be ours. The
//! filesystem already says this; asking the kernel says it *explicitly*, and
//! it is the seam a stronger check grows from.
//!
//! **The honest frame about what this does and does not protect:** same-user
//! process isolation does not exist on the desktop. Any process running as you
//! can already read the database file off disk, so this socket does not newly
//! expose the data — the bar it has to meet is *not exposing more than the
//! filesystem already does*, which is why the verb surface is typed wrappers
//! over `AppCore` methods and there is no raw-database verb (see
//! [`eidola_app_core::ipc`]). The moment something crosses it that the
//! filesystem does *not* already give away — screen-capture-derived data,
//! spend authority without a prompt — tier 0 stops being enough and a
//! code-signing check on the peer plus a per-client consent grant become
//! prerequisites, not improvements.
//!
//! ## Lifecycle
//!
//! Bound once at launch, in **both** launch modes — a windowless service is
//! precisely the process this is for, and a windowed app is one ⌘Q away from
//! being the same thing. It is deliberately **not** tied to windows or to the
//! retire: ⌘Q keeps the process, the engines and this socket, which is the
//! point of the app outliving its windows. Only a full shutdown removes the
//! file, from the same `on_app_quit` hook that drains the engines — and even
//! that is best-effort tidiness rather than correctness, since the next bind
//! replaces whatever it finds.

use std::io;
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use eidola_app_core::AppCore;
use eidola_app_core::ipc::socket_path;
use gpui::App;

use crate::stores::Stores;

/// How many consecutive `accept` failures the loop tolerates before giving up.
///
/// Some accept errors are transient and self-clearing (the process ran out of
/// file descriptors, a signal interrupted the call) — treating the first one as
/// fatal would silence the socket for the rest of a long-lived process over a
/// condition that passes. Some are not, and a listener that can only fail must
/// not spin. Retrying a bounded number of times with a pause between splits the
/// difference in the direction that keeps the door open.
const MAX_CONSECUTIVE_ACCEPT_FAILURES: u32 = 16;

/// How long the accept loop waits after a failure before trying again.
const ACCEPT_RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(100);

/// The verdict on one connecting peer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    /// Same user: serve it.
    Admit,
    /// Not ours, or not knowable. See [`Refusal`].
    Refuse(Refusal),
}

/// Why a peer was turned away.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The peer runs as a different user. The directory mode should have made
    /// this unreachable; if it happens, the directory's protection is what has
    /// failed, and this is the check that still holds.
    ForeignUser { peer_uid: u32, our_uid: u32 },
    /// The kernel would not say who the peer is. **Fail closed**: an unknown
    /// peer is not a peer we can claim is ours, and "probably fine" is not a
    /// thing an authentication check gets to say.
    UnknownPeer,
}

/// Tier-0 peer authentication, as a pure decision.
pub fn admit(peer_uid: Option<u32>, our_uid: u32) -> Admission {
    match peer_uid {
        Some(peer_uid) if peer_uid == our_uid => Admission::Admit,
        Some(peer_uid) => Admission::Refuse(Refusal::ForeignUser { peer_uid, our_uid }),
        None => Admission::Refuse(Refusal::UnknownPeer),
    }
}

/// A bound socket, held for as long as the process serves it.
pub struct ControlSocket {
    path: PathBuf,
    accepting: tokio::task::JoinHandle<()>,
}

impl ControlSocket {
    /// The path this socket is bound at.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Stop serving and remove the socket file.
    ///
    /// Best-effort by design: a bind replaces whatever it finds, so a process
    /// that dies without getting here costs the next launch nothing.
    pub fn close(&self) {
        self.accepting.abort();
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Bind the control socket and start serving it, if there is a profile to
/// serve.
///
/// Returns `None` when there is no open core (a stubbed app) or when the socket
/// could not be bound. **A failure here is not a startup failure**: the app runs
/// perfectly well without a socket — what is lost is another process's ability
/// to reach it, and the honest way to say so is a diagnostic plus a door that
/// simply is not there, not an app that refuses to start.
pub fn bind(stores: &Stores) -> Option<ControlSocket> {
    serve(&stores.app_core()?)
}

/// Bind and serve the control socket for one open core.
///
/// The half of [`bind`] that knows nothing about gpui — which is what lets the
/// socket be exercised end to end, over a real socket with a real peer, with no
/// running app around it (`tests/ipc.rs`).
pub fn serve(core: &Arc<AppCore>) -> Option<ControlSocket> {
    let path = socket_path(core.data_dir());
    let listener = match bind_listener(&path) {
        Ok(listener) => listener,
        Err(e) => {
            eprintln!("eidola-gui: not accepting local connections at {path:?}: {e}");
            return None;
        }
    };
    let accepting = core
        .runtime()
        .spawn(accept_loop(Arc::clone(core), listener));
    Some(ControlSocket { path, accepting })
}

/// Bind the control socket and arrange for a full shutdown to remove it.
///
/// The hook is registered on the same quit path that drains the engines: ⌘Q's
/// retire never reaches it, so the retired app keeps answering — which is the
/// state this socket exists to make reachable.
pub fn install(stores: &Stores, cx: &mut App) {
    let Some(socket) = bind(stores) else {
        return;
    };
    cx.on_app_quit(move |_: &mut App| {
        socket.close();
        async {}
    })
    .detach();
}

/// Create the listening socket, replacing a stale one.
fn bind_listener(path: &Path) -> io::Result<std::os::unix::net::UnixListener> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_socket() => {
            // Stale by construction — we hold the database lock, so nothing
            // else is serving this profile.
            std::fs::remove_file(path)?;
        }
        Ok(_) => {
            // Something that is not a socket is sitting on the path. Removing
            // it would be this process deleting a file it has no idea about,
            // inside the user's own data directory; refusing costs the socket
            // and nothing else.
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "the path is occupied by something that is not a socket",
            ));
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    let listener = std::os::unix::net::UnixListener::bind(path)?;
    // The mode a bind leaves behind depends on the process umask, so it is set
    // explicitly rather than inherited. The window before this line is not a
    // hole: the containing directory is `0700`, so no other user can reach the
    // path to begin with.
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    listener.set_nonblocking(true)?;
    Ok(listener)
}

/// Accept connections until the process ends.
async fn accept_loop(core: Arc<AppCore>, listener: std::os::unix::net::UnixListener) {
    let listener = match tokio::net::UnixListener::from_std(listener) {
        Ok(listener) => listener,
        Err(e) => {
            eprintln!("eidola-gui: local control socket could not start: {e}");
            return;
        }
    };
    let our_uid = our_uid();
    let mut failures = 0u32;
    loop {
        let stream = match listener.accept().await {
            Ok((stream, _addr)) => {
                failures = 0;
                stream
            }
            Err(e) => {
                failures += 1;
                if failures >= MAX_CONSECUTIVE_ACCEPT_FAILURES {
                    eprintln!("eidola-gui: local control socket stopped accepting: {e}");
                    return;
                }
                tokio::time::sleep(ACCEPT_RETRY_DELAY).await;
                continue;
            }
        };

        let peer_uid = stream.peer_cred().ok().map(|cred| cred.uid());
        if let Admission::Refuse(refusal) = admit(peer_uid, our_uid) {
            // Dropped without a word. A peer we refused to authenticate is not
            // owed a protocol frame — anything written would be this process
            // telling something it does not trust that it is here.
            drop(stream);
            match refusal {
                Refusal::ForeignUser { peer_uid, our_uid } => eprintln!(
                    "eidola-gui: refused a local connection from uid {peer_uid} (we are {our_uid})"
                ),
                Refusal::UnknownPeer => {
                    eprintln!("eidola-gui: refused a local connection with unreadable credentials")
                }
            }
            continue;
        }

        let core = Arc::clone(&core);
        tokio::spawn(async move {
            let (reader, writer) = stream.into_split();
            eidola_app_core::ipc::serve_connection(
                core,
                env!("CARGO_PKG_VERSION").to_string(),
                reader,
                writer,
            )
            .await;
        });
    }
}

/// This process's effective uid — the identity a peer has to match.
fn our_uid() -> u32 {
    // SAFETY: `geteuid` reads process state, takes no arguments and cannot
    // fail. It is unsafe only because it is `extern "C"`.
    unsafe { libc::geteuid() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_our_own_user_is_admitted() {
        assert_eq!(admit(Some(501), 501), Admission::Admit);
        assert_eq!(
            admit(Some(0), 501),
            Admission::Refuse(Refusal::ForeignUser {
                peer_uid: 0,
                our_uid: 501
            }),
            "root is another user, and this check has no notion of privilege"
        );
    }

    #[test]
    fn a_peer_the_kernel_will_not_name_is_refused() {
        assert_eq!(
            admit(None, 501),
            Admission::Refuse(Refusal::UnknownPeer),
            "an unknowable peer must fail closed"
        );
    }

    #[test]
    fn binding_replaces_a_stale_socket() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = socket_path(dir.path());
        let stale = std::os::unix::net::UnixListener::bind(&path).expect("stale socket");
        drop(stale);
        assert!(path.exists(), "the stale file outlives its listener");

        let listener = bind_listener(&path).expect("bind over the stale socket");
        assert!(path.exists());
        let mode = std::fs::metadata(&path)
            .expect("metadata")
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600, "the socket is ours alone");
        drop(listener);
    }

    #[test]
    fn binding_refuses_a_path_occupied_by_something_else() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = socket_path(dir.path());
        std::fs::write(&path, b"not a socket").expect("write");
        let err = bind_listener(&path).expect_err("a regular file is not ours to delete");
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);
        assert!(
            path.exists(),
            "the file the app did not recognise is intact"
        );
    }
}
