//! A profile its host owns (`eidola_app_core::storage`): the database, the
//! config and the update state served from memory and streamed to the host.
//!
//! - **Round trip.** A real core on files it reaches only through the host
//!   seam: a funded conversation (issuance, spends, refunds) and local posts,
//!   flushed, closed and reopened from the host's files.
//! - **Crash consistency.** Every prefix of the host's call log of a
//!   multi-transaction run, and every write in it applied halfway, reopens to
//!   a valid database at a commit boundary that keeps every flushed commit.
//! - **One thread.** Two writers contending on the browser's scheduling: the
//!   busy wait ends.
//! - **Measurements** (`--ignored`): open time, memory and write
//!   amplification of a realistic profile.

mod chat_harness;

use std::collections::BTreeMap;
use std::future::Future;
use std::io;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::time::{Duration, Instant};

use chat_harness::{MODEL, MockConfig, core_on_host, with_account};
use eidola_app_core::AppCore;
use eidola_app_core::db;
use eidola_app_core::storage::{FsHostStorage, HostFuture, HostProfile, HostStorage, ProfileImage};

// ---------------------------------------------------------------------------
// A host in memory that logs every call it applies
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum HostOp {
    Write {
        name: String,
        offset: u64,
        bytes: Vec<u8>,
    },
    Truncate {
        name: String,
        len: u64,
    },
    Sync,
    Rename {
        from: String,
        to: String,
    },
    Remove {
        name: String,
    },
}

#[derive(Default)]
struct MemHost {
    files: Mutex<BTreeMap<String, Vec<u8>>>,
    log: Mutex<Vec<HostOp>>,
}

fn apply_op(files: &mut BTreeMap<String, Vec<u8>>, op: &HostOp) {
    match op {
        HostOp::Write {
            name,
            offset,
            bytes,
        } => {
            let file = files.entry(name.clone()).or_default();
            let end = *offset as usize + bytes.len();
            if file.len() < end {
                file.resize(end, 0);
            }
            file[*offset as usize..end].copy_from_slice(bytes);
        }
        HostOp::Truncate { name, len } => {
            files
                .entry(name.clone())
                .or_default()
                .resize(*len as usize, 0);
        }
        HostOp::Sync => {}
        HostOp::Rename { from, to } => {
            if let Some(bytes) = files.remove(from) {
                files.insert(to.clone(), bytes);
            }
        }
        HostOp::Remove { name } => {
            files.remove(name);
        }
    }
}

impl MemHost {
    fn with_files(files: BTreeMap<String, Vec<u8>>) -> Arc<Self> {
        Arc::new(Self {
            files: Mutex::new(files),
            log: Mutex::new(Vec::new()),
        })
    }

    fn record(&self, op: HostOp) {
        apply_op(&mut self.files.lock().unwrap(), &op);
        self.log.lock().unwrap().push(op);
    }

    /// Apply `op` after a yield, as a real host answers after a round trip.
    fn later(&self, op: HostOp) -> HostFuture<'_, io::Result<()>> {
        Box::pin(async move {
            tokio::task::yield_now().await;
            self.record(op);
            Ok(())
        })
    }

    fn files(&self) -> BTreeMap<String, Vec<u8>> {
        self.files.lock().unwrap().clone()
    }

    fn log(&self) -> Vec<HostOp> {
        self.log.lock().unwrap().clone()
    }
}

impl HostStorage for MemHost {
    fn read(&self, name: &str) -> HostFuture<'_, io::Result<Option<Vec<u8>>>> {
        let bytes = self.files.lock().unwrap().get(name).cloned();
        Box::pin(async move { Ok(bytes) })
    }
    fn write_at(&self, name: &str, offset: u64, bytes: Vec<u8>) -> HostFuture<'_, io::Result<()>> {
        self.later(HostOp::Write {
            name: name.into(),
            offset,
            bytes,
        })
    }
    fn truncate(&self, name: &str, len: u64) -> HostFuture<'_, io::Result<()>> {
        self.later(HostOp::Truncate {
            name: name.into(),
            len,
        })
    }
    fn sync(&self, _name: &str) -> HostFuture<'_, io::Result<()>> {
        self.later(HostOp::Sync)
    }
    fn rename(&self, from: &str, to: &str) -> HostFuture<'_, io::Result<()>> {
        self.later(HostOp::Rename {
            from: from.into(),
            to: to.into(),
        })
    }
    fn remove(&self, name: &str) -> HostFuture<'_, io::Result<()>> {
        self.later(HostOp::Remove { name: name.into() })
    }
    fn list(&self) -> HostFuture<'_, io::Result<Vec<String>>> {
        let names = self.files.lock().unwrap().keys().cloned().collect();
        Box::pin(async move { Ok(names) })
    }
}

fn current_thread() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// Load `host`'s profile and open its database; the stream task is returned
/// to the caller, who runs it or drops it.
async fn open_image(
    data_dir: &Path,
    host: Arc<dyn HostStorage>,
) -> Result<(ProfileImage, HostFuture<'static, ()>, turso::Database), String> {
    let profile = HostProfile::load(data_dir.to_path_buf(), host)
        .await
        .map_err(|e| e.to_string())?;
    let (image, stream) = profile.into_parts();
    let database = db::open_on_image(&image).await.map_err(|e| e.to_string())?;
    Ok((image, stream, database))
}

// ---------------------------------------------------------------------------
// Round trip through a real core
// ---------------------------------------------------------------------------

/// A space as a reader sees it: id, title, and its messages as (role, text).
type SpaceView = (String, Option<String>, Vec<(String, String)>);

/// A host's files by name.
type Files = BTreeMap<String, Vec<u8>>;

/// What a reopened profile must still show.
#[derive(Debug, PartialEq)]
struct Visible {
    account: Option<String>,
    credentials: Vec<(String, i64, i64)>,
    spaces: Vec<SpaceView>,
}

fn visible(core: &AppCore) -> Visible {
    let rt = core.runtime();
    let mut credentials: Vec<_> = rt
        .block_on(core.wallet_credentials())
        .unwrap()
        .into_iter()
        .map(|c| (c.nonce, c.credits, c.generation))
        .collect();
    credentials.sort();
    let mut spaces = Vec::new();
    for space in rt.block_on(core.list_spaces(true)).unwrap() {
        let messages = rt
            .block_on(core.get_space_messages(space.id.clone()))
            .unwrap()
            .into_iter()
            .map(|m| (m.role, m.content))
            .collect();
        spaces.push((space.id, space.title, messages));
    }
    spaces.sort();
    Visible {
        account: core.account_fingerprint(),
        credentials,
        spaces,
    }
}

/// **A profile on a host survives a restart.** A funded conversation writes
/// the wallet (issuance, spend, refund) and the space; the account lands in
/// the config. After a flush, the core is dropped and a new one opened from
/// the same files shows all of it, through the host seam only.
#[test]
fn a_profile_on_a_host_reopens_with_its_wallet_spaces_and_account() {
    std::thread::spawn(|| {
        let dir = tempfile::tempdir().unwrap();
        let profile_dir = dir.path().join("profile");
        let host: Arc<dyn HostStorage> = Arc::new(FsHostStorage::new(profile_dir.clone()).unwrap());
        let profile = current_thread()
            .block_on(HostProfile::load(profile_dir.clone(), host.clone()))
            .unwrap();
        let (_mock, core) = core_on_host(MockConfig::default(), profile);
        with_account(&core);
        let rt = core.runtime();
        rt.block_on(core.chat("How do tides work?".into(), MODEL.into(), None))
            .expect("a funded turn");
        let answered = rt
            .block_on(core.chat("And the moon?".into(), MODEL.into(), None))
            .expect("a second turn");
        rt.block_on(core.post("A note to self.".into(), Some(answered.space_id.clone())))
            .expect("a local post");
        let before = visible(&core);
        assert!(before.account.is_some(), "the account is configured");
        assert!(!before.credentials.is_empty(), "the turns left credentials");
        assert_eq!(before.spaces.len(), 2);
        rt.block_on(core.flush_storage()).expect("flush");
        drop(core);

        let profile = current_thread()
            .block_on(HostProfile::load(profile_dir.clone(), host))
            .unwrap();
        let (_mock, core) = core_on_host(MockConfig::default(), profile);
        assert_eq!(visible(&core), before);
        let names: Vec<_> = std::fs::read_dir(&profile_dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        assert!(names.contains(&"eidola.db".to_string()), "{names:?}");
        assert!(names.contains(&"config.toml".to_string()), "{names:?}");
    })
    .join()
    .unwrap();
}

/// **A flush covers the config too, and one profile has one writer.** A
/// second load of a profile whose core is open is refused, and the config
/// saved before a flush is on the host's disk when it returns.
#[test]
fn a_flush_makes_the_config_durable_and_a_second_load_is_refused() {
    std::thread::spawn(|| {
        let dir = tempfile::tempdir().unwrap();
        let profile_dir = dir.path().join("profile");
        let host: Arc<dyn HostStorage> = Arc::new(FsHostStorage::new(profile_dir.clone()).unwrap());
        let profile = current_thread()
            .block_on(HostProfile::load(profile_dir.clone(), host.clone()))
            .unwrap();
        let core = AppCore::on_host(profile, None).unwrap();
        core.set_account_credentials("00000000-0000-0000-0000-000000000001".into(), "s".into())
            .unwrap();
        core.runtime().block_on(core.flush_storage()).unwrap();
        let saved = std::fs::read_to_string(profile_dir.join("config.toml")).unwrap();
        assert!(saved.contains("00000000-0000-0000-0000-000000000001"));

        let second = current_thread().block_on(HostProfile::load(profile_dir.clone(), host));
        assert!(
            matches!(
                second,
                Err(eidola_app_core::error::AppError::DatabaseInUse { .. })
            ),
            "a second writer must be refused"
        );
    })
    .join()
    .unwrap();
}

/// [`MemHost`] whose writes wait while its gate is shut, as a slow or
/// suspended host's would.
struct GatedHost {
    inner: Arc<MemHost>,
    open: tokio::sync::watch::Receiver<bool>,
}

impl GatedHost {
    fn after_gate<'a>(
        &'a self,
        call: HostFuture<'a, io::Result<()>>,
    ) -> HostFuture<'a, io::Result<()>> {
        let mut open = self.open.clone();
        Box::pin(async move {
            let _ = open.wait_for(|open| *open).await;
            call.await
        })
    }
}

impl HostStorage for GatedHost {
    fn read(&self, name: &str) -> HostFuture<'_, io::Result<Option<Vec<u8>>>> {
        self.inner.read(name)
    }
    fn write_at(&self, name: &str, offset: u64, bytes: Vec<u8>) -> HostFuture<'_, io::Result<()>> {
        self.after_gate(self.inner.write_at(name, offset, bytes))
    }
    fn truncate(&self, name: &str, len: u64) -> HostFuture<'_, io::Result<()>> {
        self.after_gate(self.inner.truncate(name, len))
    }
    fn sync(&self, name: &str) -> HostFuture<'_, io::Result<()>> {
        self.after_gate(self.inner.sync(name))
    }
    fn rename(&self, from: &str, to: &str) -> HostFuture<'_, io::Result<()>> {
        self.after_gate(self.inner.rename(from, to))
    }
    fn remove(&self, name: &str) -> HostFuture<'_, io::Result<()>> {
        self.after_gate(self.inner.remove(name))
    }
    fn list(&self) -> HostFuture<'_, io::Result<Vec<String>>> {
        self.inner.list()
    }
}

/// **No request leaves ahead of the commit it may rest on.** With the host
/// holding every write, a post is committed (in memory) and a request is
/// started: nothing reaches the server. Once the host lets the writes
/// through, the request arrives, and the host's files at that moment
/// already hold the post.
#[test]
fn a_request_waits_until_every_earlier_commit_is_durable_on_the_host() {
    std::thread::spawn(|| {
        let (gate, open) = tokio::sync::watch::channel(true);
        let mem = MemHost::with_files(BTreeMap::new());
        let host = Arc::new(GatedHost {
            inner: mem.clone(),
            open,
        });
        let dir = tempfile::tempdir().unwrap();
        let profile = current_thread()
            .block_on(HostProfile::load(dir.path().to_path_buf(), host))
            .unwrap();
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let core = Arc::new(
            AppCore::on_host_with_test_http_client(profile, reqwest::Client::new()).unwrap(),
        );

        // The server: the host's files as they stand when a request arrives.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let arrived: Arc<Mutex<Option<Files>>> = Arc::default();
        {
            let arrived = arrived.clone();
            let mem = mem.clone();
            std::thread::spawn(move || {
                for conn in listener.incoming() {
                    let mut seen = arrived.lock().unwrap();
                    if seen.is_none() {
                        *seen = Some(mem.files());
                    }
                    drop(conn);
                }
            });
        }
        let rt = core.runtime();
        with_account(&core);
        rt.block_on(core.set_base_url(format!("http://127.0.0.1:{port}")))
            .unwrap();
        rt.block_on(core.flush_storage()).unwrap();

        gate.send_replace(false);
        rt.block_on(core.post("committed before the request".into(), None))
            .expect("the post commits in memory");
        let asking = {
            let core = core.clone();
            std::thread::spawn(move || {
                let _ = core.runtime().block_on(core.account_balances());
            })
        };
        std::thread::sleep(Duration::from_millis(500));
        assert!(
            arrived.lock().unwrap().is_none(),
            "a request reached the server while the commit before it was not durable"
        );

        gate.send_replace(true);
        let deadline = Instant::now() + Duration::from_secs(20);
        let files = loop {
            if let Some(files) = arrived.lock().unwrap().clone() {
                break files;
            }
            assert!(Instant::now() < deadline, "the request never left");
            std::thread::sleep(Duration::from_millis(10));
        };
        asking.join().unwrap();

        let dir = tempfile::tempdir().unwrap();
        let found = current_thread().block_on(async {
            let (_image, _stream, database) =
                open_image(dir.path(), MemHost::with_files(files)).await.unwrap();
            let conn = database.connect().unwrap();
            let mut rows = conn
                .query(
                    "SELECT count(*) FROM content_block WHERE text_content = 'committed before the request'",
                    (),
                )
                .await
                .unwrap();
            rows.next().await.unwrap().unwrap().get::<i64>(0).unwrap()
        });
        assert_eq!(found, 1, "the host held the post when the request arrived");
    })
    .join()
    .unwrap();
}

/// **A core's requests wait on its own profile only.** Another profile open
/// in the same process, whose stream never runs, holds pending writes for
/// the whole test; this core's commit and request go through regardless.
#[test]
fn a_request_waits_on_its_own_profile_and_no_other() {
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let stalled_dir = tempfile::tempdir().unwrap();
        let (stalled, stalled_stream, stalled_db) = current_thread()
            .block_on(open_image(
                stalled_dir.path(),
                MemHost::with_files(BTreeMap::new()),
            ))
            .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let profile = current_thread()
            .block_on(HostProfile::load(
                dir.path().to_path_buf(),
                MemHost::with_files(BTreeMap::new()),
            ))
            .unwrap();
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let core = AppCore::on_host_with_test_http_client(profile, reqwest::Client::new()).unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let rt = core.runtime();
        with_account(&core);
        rt.block_on(core.set_base_url(format!("http://127.0.0.1:{port}")))
            .unwrap();
        rt.block_on(core.post("a commit before the request".into(), None))
            .unwrap();
        std::thread::spawn(move || drop(listener.accept()));
        let _ = rt.block_on(core.account_balances());
        drop((stalled, stalled_stream, stalled_db));
        let _ = done.send(());
    });
    finished
        .recv_timeout(Duration::from_secs(60))
        .expect("a request waited on another profile's stream");
}

/// [`MemHost`] that refuses every call on the config (`config.toml` and the
/// temporary it is replaced through), as a full or failing disk would.
struct ConfigRefusingHost(Arc<MemHost>);

impl ConfigRefusingHost {
    fn refuse(name: &str) -> HostFuture<'static, io::Result<()>> {
        let refused = io::Error::other(format!("no space left for {name}"));
        Box::pin(async move { Err(refused) })
    }
}

impl HostStorage for ConfigRefusingHost {
    fn read(&self, name: &str) -> HostFuture<'_, io::Result<Option<Vec<u8>>>> {
        self.0.read(name)
    }
    fn write_at(&self, name: &str, offset: u64, bytes: Vec<u8>) -> HostFuture<'_, io::Result<()>> {
        if name.starts_with("config.toml") {
            return Self::refuse(name);
        }
        self.0.write_at(name, offset, bytes)
    }
    fn truncate(&self, name: &str, len: u64) -> HostFuture<'_, io::Result<()>> {
        if name.starts_with("config.toml") {
            return Self::refuse(name);
        }
        self.0.truncate(name, len)
    }
    fn sync(&self, name: &str) -> HostFuture<'_, io::Result<()>> {
        self.0.sync(name)
    }
    fn rename(&self, from: &str, to: &str) -> HostFuture<'_, io::Result<()>> {
        if to.starts_with("config.toml") {
            return Self::refuse(to);
        }
        self.0.rename(from, to)
    }
    fn remove(&self, name: &str) -> HostFuture<'_, io::Result<()>> {
        self.0.remove(name)
    }
    fn list(&self) -> HostFuture<'_, io::Result<Vec<String>>> {
        self.0.list()
    }
}

/// **An account is created only once its secret is on the host.** On a
/// profile its host owns, saving the config queues it; a host that then
/// fails the write has kept nothing. The server holds only the secret's
/// hash, so the creation must hand the credential back as unsaved — what a
/// config that could not be written on the desktop does — and never report
/// an account that the next launch will not have.
#[test]
fn an_account_the_host_failed_to_keep_is_handed_back_as_unsaved() {
    std::thread::spawn(|| {
        let mem = MemHost::with_files(BTreeMap::new());
        let host: Arc<dyn HostStorage> = Arc::new(ConfigRefusingHost(mem.clone()));
        let dir = tempfile::tempdir().unwrap();
        let profile = current_thread()
            .block_on(HostProfile::load(dir.path().to_path_buf(), host))
            .unwrap();
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let core = AppCore::on_host_with_test_http_client(profile, reqwest::Client::new()).unwrap();
        let rt = core.runtime();
        let server = rt.block_on(async {
            let server = wiremock::MockServer::start().await;
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .and(wiremock::matchers::path("/v1/account"))
                .respond_with(wiremock::ResponseTemplate::new(201).set_body_json(
                    serde_json::json!({
                        "account_id": "00000000-0000-0000-0000-00000000000d",
                        "secret": "only-copy-of-the-new-secret",
                        "created_at": "2025-01-01T00:00:00Z",
                    }),
                ))
                .mount(&server)
                .await;
            wiremock::Mock::given(wiremock::matchers::method("GET"))
                .and(wiremock::matchers::path("/v1/terms"))
                .respond_with(
                    wiremock::ResponseTemplate::new(200)
                        .set_body_json(serde_json::json!({ "documents": [] })),
                )
                .mount(&server)
                .await;
            server
        });
        rt.block_on(core.set_base_url(server.uri())).unwrap();

        match rt.block_on(core.account_create(Vec::new())) {
            Err(eidola_app_core::error::AppError::AccountNotSaved { account, reason }) => {
                assert_eq!(account.account_id, "00000000-0000-0000-0000-00000000000d");
                assert_eq!(account.secret.expose(), "only-copy-of-the-new-secret");
                assert_eq!(
                    account.recovery_file, None,
                    "a recovery file the host cannot keep was named"
                );
                assert!(
                    reason.contains("could not be saved"),
                    "the reason does not say the device kept nothing: {reason}"
                );
            }
            other => panic!("an account the host never kept was reported created: {other:?}"),
        }
        drop(core);
        assert!(
            !mem.files().contains_key("config.toml"),
            "the host kept the config after all"
        );
    })
    .join()
    .unwrap();
}

// ---------------------------------------------------------------------------
// Crash consistency
// ---------------------------------------------------------------------------

const ROWS_PER_TXN: i64 = 3;
const TXNS: i64 = 36;

fn payload(seq: i64, row: i64, version: i64) -> String {
    format!("{seq}:{row}:{version};").repeat(1400 / 8)
}

/// Transaction `seq` inserts its rows and rewrites the rows of `seq - 1`, so a
/// half-applied transaction shows as one without the other.
async fn probe_txn(conn: &turso::Connection, seq: i64) {
    conn.execute("BEGIN IMMEDIATE", ()).await.unwrap();
    for row in 0..ROWS_PER_TXN {
        conn.execute(
            "INSERT INTO crash_probe (seq, row, payload) VALUES (?1, ?2, ?3)",
            (seq, row, payload(seq, row, 0)),
        )
        .await
        .unwrap();
        if seq > 1 {
            conn.execute(
                "UPDATE crash_probe SET payload = ?1 WHERE seq = ?2 AND row = ?3",
                (payload(seq - 1, row, 1), seq - 1, row),
            )
            .await
            .unwrap();
        }
    }
    conn.execute("COMMIT", ()).await.unwrap();
}

/// The number of committed transactions `database` holds, or why it is not a
/// database at a commit boundary.
async fn committed(database: &turso::Database) -> Result<i64, String> {
    let conn = db::connect(database).await.map_err(|e| e.to_string())?;
    let mut check = conn
        .query("PRAGMA integrity_check", ())
        .await
        .map_err(|e| e.to_string())?;
    let verdict: String = check
        .next()
        .await
        .map_err(|e| e.to_string())?
        .ok_or("no integrity verdict")?
        .get(0)
        .map_err(|e| e.to_string())?;
    if verdict != "ok" {
        return Err(format!("integrity: {verdict}"));
    }
    drop(check);
    let mut rows = conn
        .query(
            "SELECT seq, row, payload FROM crash_probe ORDER BY seq, row",
            (),
        )
        .await
        .map_err(|e| e.to_string())?;
    let mut found = Vec::new();
    while let Some(row) = rows.next().await.map_err(|e| e.to_string())? {
        let seq: i64 = row.get(0).map_err(|e| e.to_string())?;
        let r: i64 = row.get(1).map_err(|e| e.to_string())?;
        let p: String = row.get(2).map_err(|e| e.to_string())?;
        found.push((seq, r, p));
    }
    let m = found.last().map_or(0, |(seq, _, _)| *seq);
    let mut expected = Vec::new();
    for seq in 1..=m {
        for row in 0..ROWS_PER_TXN {
            let version = i64::from(seq < m);
            expected.push((seq, row, payload(seq, row, version)));
        }
    }
    if found != expected {
        return Err(format!(
            "not a commit boundary: {} rows for {m} transactions",
            found.len()
        ));
    }
    Ok(m)
}

struct Recorded {
    base: BTreeMap<String, Vec<u8>>,
    log: Vec<HostOp>,
    /// `(log length when the flush returned, transactions committed before
    /// it)`.
    flushes: Vec<(usize, i64)>,
}

/// A run of [`TXNS`] transactions with checkpoints and flushes between them,
/// recorded from a base that holds the initialized schema and the probe table.
fn record_run() -> Recorded {
    let dir = tempfile::tempdir().unwrap();
    current_thread().block_on(async {
        let host = MemHost::with_files(BTreeMap::new());
        let (image, stream, database) = open_image(dir.path(), host.clone()).await.unwrap();
        let stream = tokio::spawn(stream);
        let conn = db::connect(&database).await.unwrap();
        conn.execute(
            "CREATE TABLE crash_probe (seq INTEGER, row INTEGER, payload TEXT, PRIMARY KEY (seq, row))",
            (),
        )
        .await
        .unwrap();
        image.flush().await.unwrap();
        let base = host.files();
        let base_len = host.log().len();

        let mut flushes = Vec::new();
        for seq in 1..=TXNS {
            probe_txn(&conn, seq).await;
            if seq % 5 == 0 {
                let mut rows = conn.query("PRAGMA wal_checkpoint(PASSIVE)", ()).await.unwrap();
                while rows.next().await.unwrap().is_some() {}
            }
            if seq % 9 == 0 {
                let mut rows = conn.query("PRAGMA wal_checkpoint(TRUNCATE)", ()).await.unwrap();
                while rows.next().await.unwrap().is_some() {}
            }
            if seq % 4 == 0 {
                image.flush().await.unwrap();
                flushes.push((host.log().len() - base_len, seq));
            }
        }
        image.flush().await.unwrap();
        flushes.push((host.log().len() - base_len, TXNS));
        let log = host.log()[base_len..].to_vec();
        drop(conn);
        drop(database);
        drop(image);
        stream.await.unwrap();
        Recorded {
            base,
            log,
            flushes,
        }
    })
}

/// Reopen the files `base` plus `log[..k]` (plus half of `log[k]` when
/// `torn`) and count the committed transactions found.
fn reopen_prefix(rec: &Recorded, k: usize, torn: bool) -> Result<i64, String> {
    let mut files = rec.base.clone();
    for op in &rec.log[..k] {
        apply_op(&mut files, op);
    }
    if torn
        && let Some(HostOp::Write {
            name,
            offset,
            bytes,
        }) = rec.log.get(k)
    {
        let half = HostOp::Write {
            name: name.clone(),
            offset: *offset,
            bytes: bytes[..bytes.len() / 2].to_vec(),
        };
        apply_op(&mut files, &half);
    }
    let dir = tempfile::tempdir().unwrap();
    current_thread().block_on(async {
        let (_image, _stream, database) =
            open_image(dir.path(), MemHost::with_files(files)).await?;
        committed(&database).await
    })
}

/// **A crash at any point reopens to a commit boundary that keeps every
/// flushed commit.** Each prefix of the host's calls (and each write in it
/// cut in half) is reopened from scratch.
#[test]
fn every_crash_point_reopens_at_a_commit_boundary_keeping_what_was_flushed() {
    let rec = record_run();
    let writes = rec
        .log
        .iter()
        .filter(|op| matches!(op, HostOp::Write { .. }))
        .count();
    assert!(
        writes > 20,
        "the run must stream real work ({writes} writes)"
    );
    assert!(
        rec.log
            .iter()
            .any(|op| matches!(op, HostOp::Truncate { .. })),
        "the run must reset the log"
    );
    let mut last = 0;
    for k in 0..=rec.log.len() {
        for torn in [false, true] {
            if torn && !matches!(rec.log.get(k), Some(HostOp::Write { .. })) {
                continue;
            }
            let m = reopen_prefix(&rec, k, torn)
                .unwrap_or_else(|e| panic!("crash after {k} host calls (torn: {torn}): {e}"));
            let flushed = rec
                .flushes
                .iter()
                .filter(|(at, _)| *at <= k)
                .map(|(_, seq)| *seq)
                .max()
                .unwrap_or(0);
            assert!(
                m >= flushed,
                "crash after {k} host calls lost a flushed commit: found {m}, flushed {flushed}"
            );
            if !torn {
                assert!(m >= last, "a later crash point found less ({m} < {last})");
                last = m;
            }
        }
    }
    assert_eq!(last, TXNS, "the whole log reopens to every transaction");
}

/// **A first open killed at any point reopens.** The schema and its version
/// commit together, so every prefix of a first open's host calls is either
/// an empty database (made again on the next open) or the whole schema.
#[test]
fn a_first_open_killed_at_any_point_reopens() {
    let dir = tempfile::tempdir().unwrap();
    let host = MemHost::with_files(BTreeMap::new());
    current_thread().block_on(async {
        let (image, stream, database) = open_image(dir.path(), host.clone()).await.unwrap();
        let stream = tokio::spawn(stream);
        image.flush().await.unwrap();
        drop(database);
        drop(image);
        stream.await.unwrap();
    });
    let log = host.log();
    assert!(!log.is_empty());
    for k in 0..=log.len() {
        let mut files = BTreeMap::new();
        for op in &log[..k] {
            apply_op(&mut files, op);
        }
        let dir = tempfile::tempdir().unwrap();
        drop(
            current_thread()
                .block_on(open_image(dir.path(), MemHost::with_files(files)))
                .unwrap_or_else(|e| panic!("a first open killed after {k} host calls: {e}")),
        );
    }
}

// ---------------------------------------------------------------------------
// One thread, two writers
// ---------------------------------------------------------------------------

/// The browser's scheduling, natively: woken tasks run as microtasks, and a
/// timer (a macrotask) fires only once no task is ready.
#[derive(Default)]
struct MicrotaskExecutor {
    ready: Mutex<std::collections::VecDeque<usize>>,
    timers: Mutex<Vec<(Instant, Waker)>>,
}

struct TaskWaker {
    id: usize,
    exec: Arc<MicrotaskExecutor>,
}

impl Wake for TaskWaker {
    fn wake(self: Arc<Self>) {
        self.exec.ready.lock().unwrap().push_back(self.id);
    }
}

thread_local! {
    static EXEC: std::cell::RefCell<Option<Arc<MicrotaskExecutor>>> = const { std::cell::RefCell::new(None) };
}

/// A `setTimeout` on [`MicrotaskExecutor`], set when first polled.
struct Timer(Duration, Option<Instant>);

impl Future for Timer {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let delay = self.0;
        let at = *self.1.get_or_insert_with(|| Instant::now() + delay);
        if Instant::now() >= at {
            return Poll::Ready(());
        }
        EXEC.with(|e| {
            e.borrow()
                .as_ref()
                .unwrap()
                .timers
                .lock()
                .unwrap()
                .push((at, cx.waker().clone()))
        });
        Poll::Pending
    }
}

impl MicrotaskExecutor {
    fn run(self: &Arc<Self>, tasks: Vec<Pin<Box<dyn Future<Output = ()>>>>) {
        EXEC.with(|e| *e.borrow_mut() = Some(self.clone()));
        let mut tasks: Vec<Option<_>> = tasks.into_iter().map(Some).collect();
        self.ready.lock().unwrap().extend(0..tasks.len());
        loop {
            let next = self.ready.lock().unwrap().pop_front();
            if let Some(id) = next {
                let Some(task) = tasks[id].as_mut() else {
                    continue;
                };
                let waker = Waker::from(Arc::new(TaskWaker {
                    id,
                    exec: self.clone(),
                }));
                if task
                    .as_mut()
                    .poll(&mut Context::from_waker(&waker))
                    .is_ready()
                {
                    tasks[id] = None;
                }
                continue;
            }
            if tasks.iter().all(Option::is_none) {
                return;
            }
            let mut timers = self.timers.lock().unwrap();
            let Some(i) = (0..timers.len()).min_by_key(|&i| timers[i].0) else {
                panic!("every task waits and no timer is set");
            };
            let (at, waker) = timers.swap_remove(i);
            drop(timers);
            std::thread::sleep(at.saturating_duration_since(Instant::now()));
            waker.wake();
        }
    }
}

/// Run writers `a` (which holds its write transaction across `hold`, then
/// commits) and `b` (which begins once `a` holds it), on one thread with the
/// browser's scheduling. Returns how long `b` took and how it ended.
fn contend(hold: impl Future<Output = ()> + 'static) -> (Duration, Result<(), String>, Duration) {
    let dir = tempfile::tempdir().unwrap();
    let (image, stream, database) = current_thread()
        .block_on(open_image(dir.path(), MemHost::with_files(BTreeMap::new())))
        .unwrap();
    drop(stream);
    let database = Arc::new(database);
    let a_holds = Arc::new(tokio::sync::Notify::new());
    let outcome = Arc::new(Mutex::new(None));
    let a = {
        let database = database.clone();
        let a_holds = a_holds.clone();
        Box::pin(async move {
            let conn = db::connect(&database).await.unwrap();
            conn.execute("CREATE TABLE IF NOT EXISTS t (v INTEGER)", ())
                .await
                .unwrap();
            conn.execute("BEGIN IMMEDIATE", ()).await.unwrap();
            conn.execute("INSERT INTO t VALUES (1)", ()).await.unwrap();
            a_holds.notify_one();
            hold.await;
            conn.execute("INSERT INTO t VALUES (2)", ()).await.unwrap();
            conn.execute("COMMIT", ()).await.unwrap();
        }) as Pin<Box<dyn Future<Output = ()>>>
    };
    let b = {
        let database = database.clone();
        let outcome = outcome.clone();
        Box::pin(async move {
            let conn = db::connect(&database).await.unwrap();
            a_holds.notified().await;
            let started = Instant::now();
            let result = async {
                conn.execute("BEGIN IMMEDIATE", ()).await?;
                conn.execute("INSERT INTO t VALUES (3)", ()).await?;
                conn.execute("COMMIT", ()).await
            }
            .await
            .map(|_| ())
            .map_err(|e| e.to_string());
            *outcome.lock().unwrap() = Some((started.elapsed(), result));
        }) as Pin<Box<dyn Future<Output = ()>>>
    };
    let started = Instant::now();
    Arc::new(MicrotaskExecutor::default()).run(vec![a, b]);
    let total = started.elapsed();
    drop(image);
    let (took, result) = outcome.lock().unwrap().take().unwrap();
    (took, result, total)
}

/// **A writer holding the database across database work only is waited out.**
/// The holder's statements complete in memory, so the waiting writer's busy
/// loop (which wakes itself at once) still lets it run, and both commit.
#[test]
fn a_contending_writer_on_one_thread_waits_out_a_holder_doing_database_work() {
    let (took, result, _) = contend(async {
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }
    });
    assert_eq!(result, Ok(()));
    assert!(took < Duration::from_secs(1), "waited {took:?}");
}

/// **A writer holding the database across a timer stalls the page until the
/// busy timeout.** The waiting writer re-wakes itself as a microtask, so the
/// holder's timer (a macrotask) cannot fire until the wait gives up; the
/// spin is bounded by `busy_timeout`, then the waiter is refused and the
/// holder commits. On wasm32 no transaction may be held across anything but
/// database work (`crates/eidola-app-core/AGENTS.md` → Storage).
#[test]
fn a_contending_writer_on_one_thread_is_refused_at_the_busy_timeout_when_the_holder_waits_on_a_timer()
 {
    let (took, result, total) = contend(Timer(Duration::from_millis(20), None));
    eprintln!("CONTEND timer: took {took:?} total {total:?} {result:?}");
    assert!(
        result
            .as_ref()
            .is_err_and(|e| e.contains("locked") || e.contains("busy")),
        "{result:?}"
    );
    assert!(
        took >= Duration::from_secs(4) && took < Duration::from_secs(10),
        "the spin is bounded by busy_timeout: {took:?}"
    );
    assert!(total < Duration::from_secs(10));
}

// ---------------------------------------------------------------------------
// Measurements
// ---------------------------------------------------------------------------

fn rss_kib() -> u64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .unwrap_or(0)
}

/// Build a profile of about `target_mb` of spaces and posts through a real
/// core on a host, then measure the stream and a cold open.
fn measure(target_mb: u64) {
    std::thread::spawn(move || {
        let dir = tempfile::tempdir().unwrap();
        let profile_dir = dir.path().join("profile");
        let host: Arc<dyn HostStorage> = Arc::new(FsHostStorage::new(profile_dir.clone()).unwrap());
        let profile = current_thread()
            .block_on(HostProfile::load(profile_dir.clone(), host.clone()))
            .unwrap();
        let image = profile.image().clone();
        let core = AppCore::on_host(profile, None).unwrap();
        let rt = core.runtime();
        let text: String = "The tide rises and falls twice a day, pulled by the moon. ".repeat(70); // ~4 KB a post
        let mut posts = 0u64;
        let mut space = None;
        let started = Instant::now();
        while image_bytes(&profile_dir) < target_mb << 20 {
            for _ in 0..200 {
                if posts.is_multiple_of(40) {
                    space = Some(rt.block_on(core.create_space(None)).unwrap().id);
                }
                rt.block_on(core.post(format!("{posts}: {text}"), space.clone()))
                    .unwrap();
                posts += 1;
            }
            rt.block_on(core.flush_storage()).unwrap();
        }
        let write_secs = started.elapsed().as_secs_f64();
        let stats = image.stream_stats();
        let logical = posts * text.len() as u64;
        drop(image);
        drop(core);

        let db_bytes = std::fs::metadata(profile_dir.join("eidola.db"))
            .unwrap()
            .len();
        let wal_bytes = std::fs::metadata(profile_dir.join("eidola.db-wal"))
            .map(|m| m.len())
            .unwrap_or(0);
        let rss_before = rss_kib();
        let started = Instant::now();
        let profile = current_thread()
            .block_on(HostProfile::load(profile_dir.clone(), host))
            .unwrap();
        let loaded = started.elapsed();
        let core = AppCore::on_host(profile, None).unwrap();
        core.runtime().block_on(core.open_database()).unwrap();
        let db_opened = started.elapsed();
        let spaces = core.runtime().block_on(core.list_spaces(false)).unwrap();
        let opened = started.elapsed();
        let rss_after = rss_kib();
        drop(core);

        // The same files opened from the filesystem, as the desktop does.
        let disk_dir = dir.path().join("disk");
        std::fs::create_dir_all(&disk_dir).unwrap();
        for name in ["eidola.db", "eidola.db-wal"] {
            let _ = std::fs::copy(profile_dir.join(name), disk_dir.join(name));
        }
        let started = Instant::now();
        let core = AppCore::new(disk_dir.clone(), disk_dir).unwrap();
        core.runtime().block_on(core.open_database()).unwrap();
        let disk_db_opened = started.elapsed();
        core.runtime().block_on(core.list_spaces(false)).unwrap();
        let disk_opened = started.elapsed();
        drop(core);
        eprintln!(
            "MEASURE {target_mb} MB opens: image db {db_opened:?}, +list {opened:?}; \
             disk db {disk_db_opened:?}, +list {disk_opened:?}"
        );
        eprintln!(
            "MEASURE {target_mb} MB: posts {posts} in {} spaces, written in {write_secs:.1}s; \
             db {:.1} MB + wal {:.1} MB; load {loaded:?}, open+list {opened:?}; \
             rss +{:.1} MB (from {:.1} MB); stream: {} writes / {:.1} MB from turso, \
             {} host calls / {:.1} MB to host, {} syncs; logical post text {:.1} MB, \
             host bytes per post byte {:.2}",
            spaces.len(),
            db_bytes as f64 / 1048576.0,
            wal_bytes as f64 / 1048576.0,
            (rss_after.saturating_sub(rss_before)) as f64 / 1024.0,
            rss_before as f64 / 1024.0,
            stats.writes,
            stats.bytes_written as f64 / 1048576.0,
            stats.host_calls,
            stats.host_bytes as f64 / 1048576.0,
            stats.host_syncs,
            logical as f64 / 1048576.0,
            stats.host_bytes as f64 / logical as f64,
        );
    })
    .join()
    .unwrap();
}

fn image_bytes(dir: &Path) -> u64 {
    std::fs::metadata(dir.join("eidola.db"))
        .map(|m| m.len())
        .unwrap_or(0)
}

#[test]
#[ignore = "measurement: cargo test --test host_storage -- --ignored --nocapture"]
fn measure_a_20_mb_profile() {
    measure(20);
}

#[test]
#[ignore = "measurement: cargo test --test host_storage -- --ignored --nocapture"]
fn measure_a_100_mb_profile() {
    measure(100);
}
