//! Storage the embedding app owns: the host seam, and the in-memory image of
//! a profile that lives in the host's files.
//!
//! The desktop keeps its profile in ordinary files (`config.toml`, the
//! database, `update-state.json`) and reads and writes them directly. The
//! mobile app cannot: in a webview nothing durable may sit in browser storage
//! (the engine evicts it), so the profile lives in files the native shell owns
//! and reaches through an asynchronous API — [`HostStorage`]. turso cannot run
//! on an asynchronous file API, because it waits for its own reads and writes
//! synchronously, and on one thread that wait would never end. So a profile
//! opened on a host is **loaded whole into memory** ([`HostProfile::load`]),
//! every read and write turso and the config make is served from that image
//! at once, and every write is **streamed to the host in order** behind them.
//!
//! # The contract
//!
//! - **Order.** The host receives exactly the writes, truncations and syncs
//!   turso made, in the order it made them, one call at a time (contiguous
//!   writes to one file may be merged into one call). A config or
//!   update-state save is one atomic replacement in the same stream.
//! - **What the host holds after the app dies.** A prefix of that stream, of
//!   which only the last call may be partly applied. turso's write-ahead log
//!   is checksummed frame by frame and commits by a frame that marks the
//!   commit, the database file is written only by checkpoints that follow a
//!   synced log, and the log is reset only after the database is synced, so
//!   every such prefix reopens to the database as of some committed
//!   transaction, never a torn one. If the device itself loses power, the host
//!   holds what its own syncs made durable, which is what a disk holds for
//!   the desktop: the syncs are turso's, in turso's order.
//! - **Flush.** [`ProfileImage::flush`] returns `Ok` once everything written
//!   before the call is applied by the host and synced. A transaction
//!   committed before a flush that returned `Ok` survives anything that
//!   follows. The core's durability barrier (`crate::durability`) awaits it
//!   before a request leaves the device, so nothing the network learns (an issuance's request, a spend)
//!   can outrun the local record that makes it recoverable.
//! - **Failure.** A host call that fails stops the stream for good: the image
//!   keeps serving reads and writes in memory, but every later flush fails,
//!   so no request can leave on state that will not survive. Reopening loads
//!   whatever the host holds.
//! - **One writer.** [`HostProfile::load`] takes the profile's lock
//!   ([`crate::db::DbLock`]) before it reads anything, and the stream holds it
//!   until its last write is applied, so a profile reopened in the same
//!   process never loads while an earlier stream is still writing.
//!
//! Between flushes, writes are durable as soon as the stream reaches them,
//! usually within moments; a crash in that window loses only work nothing
//! outside the device has seen.

mod image;

use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use crate::config::Config;
use crate::error::AppError;
use crate::updates;

pub use image::{ProfileImage, StreamStats};

/// A host call's future. `Send` natively, where the stream runs on tokio's
/// multi-threaded runtime; on wasm32 the browser's futures are not `Send`.
#[cfg(not(target_family = "wasm"))]
pub type HostFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
#[cfg(target_family = "wasm")]
pub type HostFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

/// The bounds a host carries: `Send + Sync` natively, none on wasm32.
#[cfg(not(target_family = "wasm"))]
pub trait HostBounds: Send + Sync {}
#[cfg(not(target_family = "wasm"))]
impl<T: Send + Sync + ?Sized> HostBounds for T {}
#[cfg(target_family = "wasm")]
pub trait HostBounds {}
#[cfg(target_family = "wasm")]
impl<T: ?Sized> HostBounds for T {}

/// A directory of files the embedding app owns, holding one profile.
///
/// Names are plain file names (no separators). Every call is applied in full
/// before its future resolves, and the stream never issues a call before the
/// previous one resolved, so a host may apply them in arrival order.
pub trait HostStorage: HostBounds + 'static {
    /// The whole file, or `None` when there is none.
    fn read(&self, name: &str) -> HostFuture<'_, io::Result<Option<Vec<u8>>>>;

    /// Write `bytes` at `offset`, creating the file when it is missing and
    /// extending it when the write ends past its end (a gap reads as zeros).
    fn write_at(&self, name: &str, offset: u64, bytes: Vec<u8>) -> HostFuture<'_, io::Result<()>>;

    /// Set the file's length, creating it when it is missing.
    fn truncate(&self, name: &str, len: u64) -> HostFuture<'_, io::Result<()>>;

    /// Make the file's contents, its length and its existence durable. A
    /// missing file is not an error.
    fn sync(&self, name: &str) -> HostFuture<'_, io::Result<()>>;

    /// Replace `to` with `from` atomically, durably once it resolves.
    fn rename(&self, from: &str, to: &str) -> HostFuture<'_, io::Result<()>>;

    /// Delete the file, durably once it resolves. A missing file is not an
    /// error.
    fn remove(&self, name: &str) -> HostFuture<'_, io::Result<()>>;

    /// The names of every file in the directory.
    fn list(&self) -> HostFuture<'_, io::Result<Vec<String>>>;
}

/// A profile loaded from a host, ready for [`crate::AppCore::on_host`].
pub struct HostProfile {
    pub(crate) data_dir: PathBuf,
    pub(crate) image: ProfileImage,
    pub(crate) stream: Option<HostFuture<'static, ()>>,
}

impl HostProfile {
    /// Take the profile's lock, then load every file `host` holds into
    /// memory. `data_dir` names the profile (the lock is keyed by it, and
    /// [`crate::AppCore::data_dir`] reports it); nothing is read from it.
    pub async fn load(data_dir: PathBuf, host: Arc<dyn HostStorage>) -> Result<Self, AppError> {
        let lock = crate::db::DbLock::acquire(&data_dir, crate::db::LockHolder::App)?;
        let (image, stream) = ProfileImage::load(host, lock).await?;
        Ok(Self {
            data_dir,
            image,
            stream: Some(stream),
        })
    }

    /// The image this profile is served from.
    pub fn image(&self) -> &ProfileImage {
        &self.image
    }

    /// The image and the task that streams its writes to the host, for a
    /// caller that drives the database without an [`crate::AppCore`] (which
    /// otherwise spawns the task itself). Nothing reaches the host, and no
    /// flush returns, until the task runs.
    pub fn into_parts(mut self) -> (ProfileImage, HostFuture<'static, ()>) {
        let stream = self.stream.take().expect("taken only here");
        (self.image, stream)
    }
}

/// Where a core's profile files live: the filesystem (the desktop), or an
/// image of files its host owns. Every read and write of the config, the
/// update state and the database goes through here, so the two never mix.
pub(crate) enum ProfileFiles {
    #[cfg(not(target_family = "wasm"))]
    Disk,
    Host(ProfileImage),
}

/// The name a profile path has inside an image: its file name.
fn image_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

impl ProfileFiles {
    /// [`Config::load_from`]: defaults when missing or unparseable.
    pub(crate) fn load_config(&self, path: &Path) -> Config {
        match self {
            #[cfg(not(target_family = "wasm"))]
            Self::Disk => Config::load_from(path),
            Self::Host(image) => image
                .read(&image_name(path))
                .and_then(|bytes| String::from_utf8(bytes).ok())
                .map(|text| Config::parse_lenient(&text))
                .unwrap_or_default(),
        }
    }

    /// [`Config::load_for_update`]: defaults when missing, an error when
    /// unreadable.
    pub(crate) fn load_config_for_update(&self, path: &Path) -> Result<Config, AppError> {
        match self {
            #[cfg(not(target_family = "wasm"))]
            Self::Disk => Config::load_for_update(path),
            Self::Host(image) => match image.read(&image_name(path)) {
                None => Ok(Config::default()),
                Some(bytes) => {
                    let text = String::from_utf8(bytes).map_err(|_| AppError::Config {
                        message: format!(
                            "{} is not valid UTF-8; fix or move it aside (nothing was written)",
                            path.display()
                        ),
                    })?;
                    Config::parse_for_update(&text, path)
                }
            },
        }
    }

    /// [`Config::save_to`]; on a host, one atomic replacement in the stream.
    pub(crate) fn save_config(&self, config: &Config, path: &Path) -> Result<(), AppError> {
        match self {
            #[cfg(not(target_family = "wasm"))]
            Self::Disk => config.save_to(path),
            Self::Host(image) => image.replace(&image_name(path), config.to_toml()?.into_bytes()),
        }
    }

    /// Wait until every write made before the call is durable, within the
    /// durability barrier's bound: at once on the filesystem, whose writes
    /// are durable when they return; on a host, its flush
    /// ([`ProfileImage::flush`]), failed when the stream has stopped or the
    /// host does not answer in time.
    pub(crate) async fn durable(&self) -> Result<(), AppError> {
        match self {
            #[cfg(not(target_family = "wasm"))]
            Self::Disk => Ok(()),
            Self::Host(image) => {
                crate::durability::Durability::host(image.clone())
                    .barrier()
                    .await
            }
        }
    }

    /// Remove what a crashed [`Config::save_to`] left behind. A host
    /// profile's are removed when it is loaded.
    pub(crate) fn sweep_config_temporaries(&self, path: &Path) {
        match self {
            #[cfg(not(target_family = "wasm"))]
            Self::Disk => crate::config::sweep_stale_temporaries(path),
            Self::Host(_) => {
                let _ = path;
            }
        }
    }

    /// [`updates::load_state`].
    pub(crate) fn load_update_state(&self, data_dir: &Path) -> updates::UpdateState {
        match self {
            #[cfg(not(target_family = "wasm"))]
            Self::Disk => updates::load_state(data_dir),
            Self::Host(image) => image
                .read(&image_name(&updates::state_path(data_dir)))
                .map(|bytes| updates::state_from_bytes(&bytes))
                .unwrap_or_default(),
        }
    }

    /// [`updates::save_state`].
    pub(crate) fn save_update_state(
        &self,
        data_dir: &Path,
        state: &updates::UpdateState,
    ) -> Result<(), AppError> {
        match self {
            #[cfg(not(target_family = "wasm"))]
            Self::Disk => updates::save_state(data_dir, state),
            Self::Host(image) => image.replace(
                &image_name(&updates::state_path(data_dir)),
                updates::state_to_bytes(state)?,
            ),
        }
    }

    /// [`crate::db::open`].
    pub(crate) async fn open_database(&self, data_dir: &Path) -> Result<turso::Database, AppError> {
        match self {
            #[cfg(not(target_family = "wasm"))]
            Self::Disk => crate::db::open(data_dir).await,
            Self::Host(image) => {
                let _ = data_dir;
                crate::db::open_on_image(image).await
            }
        }
    }
}

/// [`HostStorage`] over a directory of the local filesystem: what the mobile
/// shell's file API does, natively, for tests and tools.
#[cfg(not(target_family = "wasm"))]
pub struct FsHostStorage {
    root: PathBuf,
    /// Files created since the directory was last synced: syncing one also
    /// syncs the directory, so its entry is as durable as its contents.
    created: std::sync::Mutex<std::collections::HashSet<String>>,
}

#[cfg(not(target_family = "wasm"))]
impl FsHostStorage {
    /// Serve the files of `root`, creating it if needed.
    pub fn new(root: PathBuf) -> io::Result<Self> {
        std::fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            created: std::sync::Mutex::new(std::collections::HashSet::new()),
        })
    }

    fn path(&self, name: &str) -> io::Result<PathBuf> {
        if name.is_empty() || name.contains(['/', '\\']) || name == "." || name == ".." {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("not a file name: {name:?}"),
            ));
        }
        Ok(self.root.join(name))
    }

    fn open_for_write(&self, name: &str) -> io::Result<std::fs::File> {
        let path = self.path(name)?;
        let existed = path.exists();
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create(true).truncate(false);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let file = options.open(&path)?;
        if !existed {
            self.created
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(name.to_string());
        }
        Ok(file)
    }

    fn sync_dir(&self) -> io::Result<()> {
        #[cfg(unix)]
        std::fs::File::open(&self.root)?.sync_all()?;
        Ok(())
    }
}

#[cfg(not(target_family = "wasm"))]
impl HostStorage for FsHostStorage {
    fn read(&self, name: &str) -> HostFuture<'_, io::Result<Option<Vec<u8>>>> {
        let path = self.path(name);
        Box::pin(async move {
            match std::fs::read(path?) {
                Ok(bytes) => Ok(Some(bytes)),
                Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(e),
            }
        })
    }

    fn write_at(&self, name: &str, offset: u64, bytes: Vec<u8>) -> HostFuture<'_, io::Result<()>> {
        let file = self.open_for_write(name);
        Box::pin(async move {
            #[cfg(unix)]
            std::os::unix::fs::FileExt::write_all_at(&file?, &bytes, offset)?;
            #[cfg(windows)]
            {
                let file = file?;
                let mut done = 0;
                while done < bytes.len() {
                    done += std::os::windows::fs::FileExt::seek_write(
                        &file,
                        &bytes[done..],
                        offset + done as u64,
                    )?;
                }
            }
            Ok(())
        })
    }

    fn truncate(&self, name: &str, len: u64) -> HostFuture<'_, io::Result<()>> {
        let file = self.open_for_write(name);
        Box::pin(async move { file?.set_len(len) })
    }

    fn sync(&self, name: &str) -> HostFuture<'_, io::Result<()>> {
        let path = self.path(name);
        let name = name.to_string();
        Box::pin(async move {
            match std::fs::OpenOptions::new().write(true).open(path?) {
                Ok(file) => file.sync_all()?,
                Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
                Err(e) => return Err(e),
            }
            let created = self
                .created
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&name);
            if created {
                self.sync_dir()?;
            }
            Ok(())
        })
    }

    fn rename(&self, from: &str, to: &str) -> HostFuture<'_, io::Result<()>> {
        let paths = self.path(from).and_then(|f| Ok((f, self.path(to)?)));
        let from = from.to_string();
        Box::pin(async move {
            let (from_path, to_path) = paths?;
            std::fs::rename(from_path, to_path)?;
            self.created
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&from);
            self.sync_dir()
        })
    }

    fn remove(&self, name: &str) -> HostFuture<'_, io::Result<()>> {
        let path = self.path(name);
        let name = name.to_string();
        Box::pin(async move {
            match std::fs::remove_file(path?) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
                Err(e) => return Err(e),
            }
            self.created
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&name);
            self.sync_dir()
        })
    }

    fn list(&self) -> HostFuture<'_, io::Result<Vec<String>>> {
        Box::pin(async move {
            let mut names = Vec::new();
            for entry in std::fs::read_dir(&self.root)? {
                let entry = entry?;
                if entry.file_type()?.is_file()
                    && let Some(name) = entry.file_name().to_str()
                {
                    names.push(name.to_string());
                }
            }
            names.sort();
            Ok(names)
        })
    }
}
