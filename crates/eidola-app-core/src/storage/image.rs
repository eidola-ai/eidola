//! The in-memory image of a host profile, the turso IO served from it, and
//! the stream that carries its writes to the host (contract: `super`).

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use turso::core as tc;

use super::{HostFuture, HostStorage};
use crate::error::AppError;

/// The granularity a file image is held in.
const PAGE: usize = 4096;

/// The most bytes one merged host write carries.
const MAX_MERGED_WRITE: usize = 1 << 20;

/// A file a crashed replacement left behind: `<name>.tmp`.
const TMP_SUFFIX: &str = ".tmp";

/// Distinguishes images in turso's process-wide database registry, which
/// keys an open database by its files' identity.
static NEXT_IMAGE_ID: AtomicU64 = AtomicU64::new(1);

/// A file's bytes, in pages, so growing a large file never copies it.
#[derive(Default)]
struct Pages {
    pages: BTreeMap<usize, Box<[u8; PAGE]>>,
    len: u64,
}

impl Pages {
    fn from_bytes(bytes: &[u8]) -> Self {
        let mut pages = Self::default();
        pages.write(0, bytes);
        pages
    }

    fn write(&mut self, pos: u64, data: &[u8]) {
        let mut offset = pos as usize;
        let mut done = 0;
        while done < data.len() {
            let in_page = offset % PAGE;
            let n = (data.len() - done).min(PAGE - in_page);
            let page = self
                .pages
                .entry(offset / PAGE)
                .or_insert_with(|| Box::new([0; PAGE]));
            page[in_page..in_page + n].copy_from_slice(&data[done..done + n]);
            offset += n;
            done += n;
        }
        self.len = self.len.max(pos + data.len() as u64);
    }

    /// Read from `pos` into `dst`, up to the end of the file; holes read as
    /// zeros. Returns the bytes read.
    fn read(&self, pos: u64, dst: &mut [u8]) -> usize {
        if pos >= self.len {
            return 0;
        }
        let n = (dst.len() as u64).min(self.len - pos) as usize;
        let mut offset = pos as usize;
        let mut done = 0;
        while done < n {
            let in_page = offset % PAGE;
            let k = (n - done).min(PAGE - in_page);
            match self.pages.get(&(offset / PAGE)) {
                Some(page) => dst[done..done + k].copy_from_slice(&page[in_page..in_page + k]),
                None => dst[done..done + k].fill(0),
            }
            offset += k;
            done += k;
        }
        n
    }

    fn truncate(&mut self, len: u64) {
        if len < self.len {
            let keep = (len as usize).div_ceil(PAGE);
            self.pages.split_off(&keep);
            let in_page = len as usize % PAGE;
            if in_page != 0
                && let Some(page) = self.pages.get_mut(&(len as usize / PAGE))
            {
                page[in_page..].fill(0);
            }
        }
        self.len = len;
    }

    fn to_vec(&self) -> Vec<u8> {
        let mut out = vec![0; self.len as usize];
        self.read(0, &mut out);
        out
    }
}

/// One call the stream will make on the host.
enum Op {
    Write {
        file: String,
        offset: u64,
        bytes: Vec<u8>,
    },
    Truncate {
        file: String,
        len: u64,
    },
    Sync {
        file: String,
    },
    /// Replace the whole file atomically (`<file>.tmp`, synced, renamed over).
    Replace {
        file: String,
        bytes: Vec<u8>,
    },
    Remove {
        file: String,
    },
    /// Sync every file written since its last sync, then mark everything
    /// before this point durable.
    Flush,
}

impl Op {
    fn bytes(&self) -> usize {
        match self {
            Op::Write { bytes, .. } | Op::Replace { bytes, .. } => bytes.len(),
            _ => 0,
        }
    }
}

/// What the stream has done so far (see [`ProfileImage::stream_stats`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamStats {
    /// Writes turso and the config made, before merging.
    pub writes: u64,
    /// Bytes those writes carried.
    pub bytes_written: u64,
    /// Calls the stream made on the host.
    pub host_calls: u64,
    /// Bytes the stream wrote to the host.
    pub host_bytes: u64,
    /// Syncs the stream asked the host for.
    pub host_syncs: u64,
    /// Bytes waiting to be streamed now.
    pub queued_bytes: u64,
}

#[derive(Default)]
struct StreamState {
    queue: VecDeque<(u64, Op)>,
    /// The last sequence number handed out.
    last_seq: u64,
    /// Everything up to this sequence number is applied and synced.
    durable_seq: u64,
    /// Why the stream stopped, once it has.
    failed: Option<String>,
    stats: StreamStats,
}

/// The ordered queue of host calls, shared by the image, its files and the
/// task that drains it.
#[derive(Default)]
struct Stream {
    state: Mutex<StreamState>,
    /// Wakes the stream task: work was queued, or the image closed.
    wake: tokio::sync::Notify,
    /// Bumped whenever `durable_seq` moves or the stream stops.
    progress: tokio::sync::watch::Sender<u64>,
}

impl Stream {
    fn lock(&self) -> std::sync::MutexGuard<'_, StreamState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Queue `op` while `state` is held, so it lands in the stream in the
    /// same order its effect landed in memory.
    fn push(&self, state: &mut StreamState, op: Op) -> u64 {
        if let Op::Write { bytes, .. } = &op {
            state.stats.writes += 1;
            state.stats.bytes_written += bytes.len() as u64;
        }
        state.stats.queued_bytes += op.bytes() as u64;
        state.last_seq += 1;
        let seq = state.last_seq;
        state.queue.push_back((seq, op));
        self.wake.notify_one();
        seq
    }

    fn stop(&self, why: String) {
        let mut state = self.lock();
        if state.failed.is_none() {
            state.failed = Some(why);
        }
        let seq = state.durable_seq;
        drop(state);
        self.progress.send_replace(seq);
    }

    async fn flush(&self) -> Result<(), AppError> {
        let mut progress = self.progress.subscribe();
        let target = {
            let mut state = self.lock();
            if let Some(why) = &state.failed {
                return Err(stream_failed(why));
            }
            if state.last_seq == state.durable_seq {
                return Ok(());
            }
            self.push(&mut state, Op::Flush)
        };
        loop {
            {
                let state = self.lock();
                if state.durable_seq >= target {
                    return Ok(());
                }
                if let Some(why) = &state.failed {
                    return Err(stream_failed(why));
                }
            }
            if progress.changed().await.is_err() {
                return Err(stream_failed("the stream is gone"));
            }
        }
    }
}

fn stream_failed(why: &str) -> AppError {
    AppError::Database {
        message: format!("the profile could not be saved to the device: {why}"),
    }
}

/// A profile's files, held in memory and streamed to its host.
///
/// Cheap to clone; every clone is the same image.
#[derive(Clone)]
pub struct ProfileImage {
    inner: Arc<ImageInner>,
}

struct ImageInner {
    id: u64,
    files: Mutex<HashMap<String, Arc<ImageFile>>>,
    stream: Arc<Stream>,
}

impl Drop for ImageInner {
    fn drop(&mut self) {
        self.stream.wake.notify_one();
    }
}

impl ProfileImage {
    /// Load every file `host` holds into memory, and return the image with
    /// the task that streams its writes back. A replacement's temporary left
    /// by a crash is removed first, as the stream's first call.
    pub(super) async fn load(
        host: Arc<dyn HostStorage>,
        lock: crate::db::DbLock,
    ) -> Result<(Self, HostFuture<'static, ()>), AppError> {
        let load_err = |e: std::io::Error| AppError::Database {
            message: format!("failed to load the profile from the device: {e}"),
        };
        let stream = Arc::new(Stream::default());
        let mut files = HashMap::new();
        for name in host.list().await.map_err(load_err)? {
            if name.ends_with(TMP_SUFFIX) {
                let mut state = stream.lock();
                stream.push(&mut state, Op::Remove { file: name });
                continue;
            }
            let Some(bytes) = host.read(&name).await.map_err(load_err)? else {
                continue;
            };
            let file = ImageFile {
                name: name.clone(),
                pages: RwLock::new(Pages::from_bytes(&bytes)),
                stream: stream.clone(),
            };
            files.insert(name, Arc::new(file));
        }
        let inner = Arc::new(ImageInner {
            id: NEXT_IMAGE_ID.fetch_add(1, Ordering::Relaxed),
            files: Mutex::new(files),
            stream: stream.clone(),
        });
        // The guard is captured, so dropping the task even unpolled stops
        // the stream, and a flush waiting on it fails rather than waiting.
        let task = Box::pin(run_stream(Stopped(stream), host, lock));
        Ok((Self { inner }, task))
    }

    /// Returns once every write made before the call is applied by the host
    /// and synced; fails if the stream has stopped (module docs).
    pub async fn flush(&self) -> Result<(), AppError> {
        self.inner.stream.flush().await
    }

    /// Counters of the stream, for measurement.
    pub fn stream_stats(&self) -> StreamStats {
        self.inner.stream.lock().stats
    }

    /// The turso IO that serves the database from this image.
    pub(crate) fn turso_io(&self) -> Arc<dyn tc::IO> {
        Arc::new(ImageIo {
            image: self.inner.clone(),
        })
    }

    /// The whole of `name`, or `None` when there is no such file.
    pub(crate) fn read(&self, name: &str) -> Option<Vec<u8>> {
        let files = self.inner.files.lock().unwrap_or_else(|e| e.into_inner());
        let file = files.get(name)?;
        Some(
            file.pages
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .to_vec(),
        )
    }

    /// Replace `name` whole, in memory now and on the host atomically in
    /// stream order. Fails only once the stream has stopped.
    pub(crate) fn replace(&self, name: &str, bytes: Vec<u8>) -> Result<(), AppError> {
        let stream = &self.inner.stream;
        let mut state = stream.lock();
        if let Some(why) = &state.failed {
            return Err(stream_failed(why));
        }
        let file = self.file(name, true).expect("created");
        *file.pages.write().unwrap_or_else(|e| e.into_inner()) = Pages::from_bytes(&bytes);
        stream.push(
            &mut state,
            Op::Replace {
                file: name.to_string(),
                bytes,
            },
        );
        Ok(())
    }

    fn file(&self, name: &str, create: bool) -> Option<Arc<ImageFile>> {
        let mut files = self.inner.files.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(file) = files.get(name) {
            return Some(file.clone());
        }
        if !create {
            return None;
        }
        let file = Arc::new(ImageFile {
            name: name.to_string(),
            pages: RwLock::new(Pages::default()),
            stream: self.inner.stream.clone(),
        });
        files.insert(name.to_string(), file.clone());
        Some(file)
    }
}

/// The stream task's hold on the stream. Stops the stream when the task is
/// dropped, finished or not (a runtime shut down, a task never spawned), so a
/// flush waiting on it fails rather than waiting forever.
struct Stopped(Arc<Stream>);

impl Drop for Stopped {
    fn drop(&mut self) {
        self.0.stop("the stream's task is gone".into());
    }
}

/// Drain the stream into the host, in order, until the image is closed and
/// everything queued is applied. Holds the profile's lock until then.
async fn run_stream(guard: Stopped, host: Arc<dyn HostStorage>, lock: crate::db::DbLock) {
    let stream = &guard.0;
    let mut dirty = BTreeSet::new();
    loop {
        let batch: Vec<(u64, Op)> = stream.lock().queue.drain(..).collect();
        if batch.is_empty() {
            // Only this task holds the stream: the image and every file
            // turso opened are gone, and nothing more can be queued.
            if Arc::strong_count(stream) == 1 {
                break;
            }
            // A notification sent since the drain is kept as a permit, so
            // this wakes at once for work queued in between.
            stream.wake.notified().await;
            continue;
        }
        let queued: u64 = batch.iter().map(|(_, op)| op.bytes() as u64).sum();
        // Once stopped, nothing more may reach the host (it would land out of
        // order), so the rest is dropped until the image closes.
        let applied = if stream.lock().failed.is_some() {
            Ok(())
        } else {
            apply(stream, &*host, batch, &mut dirty).await
        };
        let mut state = stream.lock();
        state.stats.queued_bytes = state.stats.queued_bytes.saturating_sub(queued);
        drop(state);
        if let Err(why) = applied {
            stream.stop(why);
        }
    }
    drop(lock);
}

/// Apply one batch in order, merging contiguous writes to one file.
async fn apply(
    stream: &Stream,
    host: &dyn HostStorage,
    batch: Vec<(u64, Op)>,
    dirty: &mut BTreeSet<String>,
) -> Result<(), String> {
    let count = |calls: u64, bytes: u64, syncs: u64| {
        let mut state = stream.lock();
        state.stats.host_calls += calls;
        state.stats.host_bytes += bytes;
        state.stats.host_syncs += syncs;
    };
    let fail = |what: &str, file: &str, e: std::io::Error| format!("{what} {file}: {e}");
    let mut merged: Option<(String, u64, Vec<u8>)> = None;
    for (seq, op) in batch {
        if let Op::Write {
            file,
            offset,
            bytes,
        } = op
        {
            match &mut merged {
                Some((f, start, buf))
                    if *f == file
                        && *start + buf.len() as u64 == offset
                        && buf.len() + bytes.len() <= MAX_MERGED_WRITE =>
                {
                    buf.extend_from_slice(&bytes)
                }
                _ => {
                    if let Some((f, start, buf)) = merged.take() {
                        let n = buf.len() as u64;
                        host.write_at(&f, start, buf)
                            .await
                            .map_err(|e| fail("writing", &f, e))?;
                        count(1, n, 0);
                    }
                    dirty.insert(file.clone());
                    merged = Some((file, offset, bytes));
                }
            }
            continue;
        }
        if let Some((f, start, buf)) = merged.take() {
            let n = buf.len() as u64;
            host.write_at(&f, start, buf)
                .await
                .map_err(|e| fail("writing", &f, e))?;
            count(1, n, 0);
        }
        match op {
            Op::Write { .. } => unreachable!("handled above"),
            Op::Truncate { file, len } => {
                host.truncate(&file, len)
                    .await
                    .map_err(|e| fail("truncating", &file, e))?;
                count(1, 0, 0);
                dirty.insert(file);
            }
            Op::Sync { file } => {
                host.sync(&file)
                    .await
                    .map_err(|e| fail("syncing", &file, e))?;
                count(1, 0, 1);
                dirty.remove(&file);
            }
            Op::Replace { file, bytes } => {
                let tmp = format!("{file}{TMP_SUFFIX}");
                let n = bytes.len() as u64;
                host.truncate(&tmp, 0)
                    .await
                    .map_err(|e| fail("replacing", &file, e))?;
                host.write_at(&tmp, 0, bytes)
                    .await
                    .map_err(|e| fail("replacing", &file, e))?;
                host.sync(&tmp)
                    .await
                    .map_err(|e| fail("replacing", &file, e))?;
                host.rename(&tmp, &file)
                    .await
                    .map_err(|e| fail("replacing", &file, e))?;
                count(4, n, 1);
                dirty.remove(&file);
            }
            Op::Remove { file } => {
                host.remove(&file)
                    .await
                    .map_err(|e| fail("removing", &file, e))?;
                count(1, 0, 0);
                dirty.remove(&file);
            }
            Op::Flush => {
                for file in std::mem::take(dirty) {
                    host.sync(&file)
                        .await
                        .map_err(|e| fail("syncing", &file, e))?;
                    count(1, 0, 1);
                }
                stream.lock().durable_seq = seq;
                stream.progress.send_replace(seq);
            }
        }
    }
    if let Some((f, start, buf)) = merged.take() {
        let n = buf.len() as u64;
        host.write_at(&f, start, buf)
            .await
            .map_err(|e| fail("writing", &f, e))?;
        count(1, n, 0);
    }
    Ok(())
}

/// One file of the image, as turso sees it.
struct ImageFile {
    name: String,
    pages: RwLock<Pages>,
    stream: Arc<Stream>,
}

impl Drop for ImageFile {
    fn drop(&mut self) {
        self.stream.wake.notify_one();
    }
}

impl ImageFile {
    fn pages_mut(&self) -> std::sync::RwLockWriteGuard<'_, Pages> {
        self.pages.write().unwrap_or_else(|e| e.into_inner())
    }

    /// Apply `change` in memory and queue `op`, as one step in stream order.
    fn mutate(&self, change: impl FnOnce(&mut Pages), op: Op) {
        let mut state = self.stream.lock();
        change(&mut self.pages_mut());
        self.stream.push(&mut state, op);
    }
}

impl tc::File for ImageFile {
    fn lock_file(&self, _exclusive: bool) -> tc::Result<()> {
        // The profile's own lock keeps a second writer out.
        Ok(())
    }

    fn unlock_file(&self) -> tc::Result<()> {
        Ok(())
    }

    fn pread(&self, pos: u64, c: tc::Completion) -> tc::Result<tc::Completion> {
        let n = {
            let pages = self.pages.read().unwrap_or_else(|e| e.into_inner());
            pages.read(pos, c.as_read().buf().as_mut_slice())
        };
        c.complete(n as i32);
        Ok(c)
    }

    fn pwrite(
        &self,
        pos: u64,
        buffer: Arc<tc::Buffer>,
        c: tc::Completion,
    ) -> tc::Result<tc::Completion> {
        let bytes = buffer.as_slice().to_vec();
        let n = bytes.len();
        self.mutate(
            |pages| pages.write(pos, &bytes),
            Op::Write {
                file: self.name.clone(),
                offset: pos,
                bytes: bytes.clone(),
            },
        );
        c.complete(n as i32);
        Ok(c)
    }

    fn pwritev(
        &self,
        pos: u64,
        buffers: Vec<Arc<tc::Buffer>>,
        c: tc::Completion,
    ) -> tc::Result<tc::Completion> {
        let mut bytes = Vec::with_capacity(buffers.iter().map(|b| b.len()).sum());
        for buffer in &buffers {
            bytes.extend_from_slice(buffer.as_slice());
        }
        let n = bytes.len();
        self.mutate(
            |pages| pages.write(pos, &bytes),
            Op::Write {
                file: self.name.clone(),
                offset: pos,
                bytes: bytes.clone(),
            },
        );
        c.complete(n as i32);
        Ok(c)
    }

    fn sync(
        &self,
        c: tc::Completion,
        _sync_type: tc::io::FileSyncType,
    ) -> tc::Result<tc::Completion> {
        // Done in memory now; the host syncs in stream order.
        let mut state = self.stream.lock();
        self.stream.push(
            &mut state,
            Op::Sync {
                file: self.name.clone(),
            },
        );
        drop(state);
        c.complete(0);
        Ok(c)
    }

    fn size(&self) -> tc::Result<u64> {
        Ok(self.pages.read().unwrap_or_else(|e| e.into_inner()).len)
    }

    fn truncate(&self, len: u64, c: tc::Completion) -> tc::Result<tc::Completion> {
        self.mutate(
            |pages| pages.truncate(len),
            Op::Truncate {
                file: self.name.clone(),
                len,
            },
        );
        c.complete(0);
        Ok(c)
    }
}

/// turso's IO over an image: every call completes before it returns.
struct ImageIo {
    image: Arc<ImageInner>,
}

impl tc::Clock for ImageIo {
    fn current_time_monotonic(&self) -> tc::MonotonicInstant {
        static EPOCH: std::sync::OnceLock<crate::rt::Instant> = std::sync::OnceLock::new();
        let epoch = *EPOCH.get_or_init(crate::rt::Instant::now);
        tc::MonotonicInstant::from_nanos(epoch.elapsed().as_nanos())
    }

    fn current_time_wall_clock(&self) -> tc::WallClockInstant {
        let since = crate::rt::SystemTime::now()
            .duration_since(crate::rt::UNIX_EPOCH)
            .unwrap_or_default();
        tc::WallClockInstant {
            secs: since.as_secs() as i64,
            micros: since.subsec_micros(),
        }
    }
}

impl tc::IO for ImageIo {
    fn open_file(
        &self,
        path: &str,
        flags: tc::OpenFlags,
        _direct: bool,
    ) -> tc::Result<Arc<dyn tc::File>> {
        // A profile file is a plain name. Anything else is a temporary file
        // turso makes (natively under a temporary directory) for sorts and
        // spills: kept in memory of its own, never streamed, gone when turso
        // drops it, as turso does on wasm32.
        if path.contains(['/', '\\']) {
            return tc::MemoryIO::new().open_file(path, flags, false);
        }
        let image = ProfileImage {
            inner: self.image.clone(),
        };
        match image.file(path, flags.contains(tc::OpenFlags::Create)) {
            Some(file) => Ok(file),
            None => Err(tc::CompletionError::IOError(std::io::ErrorKind::NotFound, "open").into()),
        }
    }

    fn remove_file(&self, path: &str) -> tc::Result<()> {
        if path.contains(['/', '\\']) {
            // A temporary file (see `open_file`): it was never streamed.
            return Ok(());
        }
        let mut state = self.image.stream.lock();
        self.image
            .files
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(path);
        self.image.stream.push(
            &mut state,
            Op::Remove {
                file: path.to_string(),
            },
        );
        Ok(())
    }

    fn file_id(&self, path: &str) -> tc::Result<tc::io::FileId> {
        // Unique per image, so two images in one process (tests) never share
        // a database through turso's registry.
        let mut id = tc::io::FileId::from_path_hash(path);
        id.dev = self.image.id;
        Ok(id)
    }

    // One thread on wasm32: there is no one to yield to, and nothing to
    // sleep with (`std::thread::sleep` panics there).
    #[cfg(target_family = "wasm")]
    fn yield_now(&self) {}

    #[cfg(target_family = "wasm")]
    fn sleep(&self, _duration: std::time::Duration) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_read_back_what_was_written_and_truncate_zeroes_the_tail() {
        let mut pages = Pages::default();
        pages.write(10, &[7; 5000]);
        assert_eq!(pages.len, 5010);
        let mut buf = vec![1; 6000];
        assert_eq!(pages.read(0, &mut buf), 5010);
        assert!(buf[..10].iter().all(|&b| b == 0));
        assert!(buf[10..5010].iter().all(|&b| b == 7));
        pages.truncate(4100);
        assert_eq!(pages.to_vec().len(), 4100);
        pages.write(5000, &[1]);
        let all = pages.to_vec();
        assert!(
            all[4100..5000].iter().all(|&b| b == 0),
            "a truncated tail reads as zeros"
        );
    }
}
