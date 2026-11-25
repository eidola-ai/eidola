//! A shell written in Rust, speaking the contract's JSON: files in memory,
//! the TCP pipe over real sockets, and a bundle store with the contract's
//! staging, sealing (the digest computed from the bytes stored) and
//! provisional activation, document loads included ([`FakeShell::relaunch`],
//! [`FakeShell::load_document`]).
//! It keeps a log of every request's arrival and every reply, in order, so a
//! test can read the order in which things reached it.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use eidola_mobile::client::{BoxFuture, Transport};
use eidola_mobile::protocol::REQUIRED_METHODS;
use serde_json::{Value, json};
use sha2::Digest as _;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::net::tcp::OwnedReadHalf;
use tokio::sync::{mpsc, oneshot};

/// The shell's own threads: never dropped, so no test drops a runtime
/// inside another.
static RT: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime")
});

enum Outgoing {
    Data(
        Vec<u8>,
        oneshot::Sender<Result<Value, (&'static str, String)>>,
    ),
    Shutdown(oneshot::Sender<Result<Value, (&'static str, String)>>),
}

struct Sock {
    tx: mpsc::UnboundedSender<Outgoing>,
    rx: Arc<tokio::sync::Mutex<OwnedReadHalf>>,
}

#[derive(Default)]
pub struct State {
    pub files: BTreeMap<String, Vec<u8>>,
    /// `"-> method target"` on arrival, `"<- method target"` on reply.
    pub log: Vec<String>,
    pub syncs: Vec<String>,
    sockets: HashMap<u64, Sock>,
    pending_connects: HashMap<u64, ()>,
    pub receives_in_flight: usize,
    pub most_receives_in_flight: usize,
    /// Bytes handed to the app by `tcp.receive`, per socket.
    pub received: HashMap<u64, usize>,
    pub confirms: usize,
    /// How long a file request waits before its reply.
    pub file_delay: Duration,
    /// `files.read` hands back the file with its last byte flipped.
    pub corrupt_reads: bool,
    pub bundles: Bundles,
}

/// One stored bundle version.
#[derive(Clone, Default)]
pub struct StoredBundle {
    pub sealed: bool,
    pub files: BTreeMap<String, Vec<u8>>,
    pub digest: Option<String>,
}

/// The bundle store and its provisional-activation state
/// (`shell-contract.md` → Bundle store).
#[derive(Clone)]
pub struct Bundles {
    pub versions: BTreeMap<String, StoredBundle>,
    pub active: String,
    pub confirmed: bool,
    pub tried: bool,
    pub previous: String,
    pub failed: Option<String>,
    /// The version this document was served from, and whether it was
    /// confirmed when it was.
    pub serving: String,
    pub serving_confirmed: bool,
}

/// What the shell shows as the embedded bundle's digest.
pub const EMBEDDED_DIGEST: &str = "00";

impl Default for Bundles {
    fn default() -> Self {
        Self {
            versions: BTreeMap::new(),
            active: "embedded".into(),
            confirmed: true,
            tried: false,
            previous: "embedded".into(),
            failed: None,
            serving: "embedded".into(),
            serving_confirmed: true,
        }
    }
}

impl Bundles {
    /// A main-frame document load: the contract's three rules, then serve
    /// `active`.
    pub fn load_document(&mut self) {
        if !self.confirmed && !self.tried {
            self.tried = true;
        } else if !self.confirmed {
            self.failed = Some(self.active.clone());
            self.active = self.previous.clone();
            self.confirmed = true;
            self.previous = "embedded".into();
        }
        if self.active != "embedded"
            && !self
                .versions
                .get(&self.active)
                .is_some_and(|v| v.sealed && v.files.contains_key("index.html"))
        {
            self.failed = Some(self.active.clone());
            self.active = "embedded".into();
            self.previous = "embedded".into();
            self.confirmed = true;
        }
        self.serving = self.active.clone();
        self.serving_confirmed = self.confirmed;
    }

    fn digest_of(&self, version: &str) -> String {
        if version == "embedded" {
            return EMBEDDED_DIGEST.into();
        }
        self.versions
            .get(version)
            .and_then(|v| v.digest.clone())
            .unwrap_or_default()
    }

    fn state(&self) -> Value {
        json!({
            "serving": {
                "version": self.serving,
                "digest": self.digest_of(&self.serving),
                "confirmed": self.serving_confirmed,
            },
            "active": self.active,
            "confirmed": self.confirmed,
            "previous": self.previous,
            "failed": self.failed,
            "versions": self.versions.iter().map(|(name, v)| json!({
                "version": name, "sealed": v.sealed, "digest": v.digest,
            })).collect::<Vec<_>>(),
        })
    }
}

/// The shell. Clones share one state.
#[derive(Clone)]
pub struct FakeShell {
    pub state: Arc<Mutex<State>>,
    contract: u64,
    methods: Vec<String>,
    /// Identifies a profile across relaunches (shared with every relaunch).
    profile: Arc<()>,
    /// Every reply is this text instead (a broken shell).
    garbage: Option<&'static str>,
}

impl Default for FakeShell {
    fn default() -> Self {
        Self {
            state: Arc::default(),
            contract: 2,
            methods: REQUIRED_METHODS.iter().map(|m| m.to_string()).collect(),
            profile: Arc::new(()),
            garbage: None,
        }
    }
}

type Reply = Result<Value, (&'static str, String)>;

fn err(code: &'static str, message: impl Into<String>) -> Reply {
    Err((code, message.into()))
}

fn bytes(params: &Value, key: &str) -> Result<Vec<u8>, (&'static str, String)> {
    let text = params[key]
        .as_str()
        .ok_or(("bad_request", key.to_string()))?;
    STANDARD
        .decode(text)
        .map_err(|e| ("bad_request", e.to_string()))
}

fn int(params: &Value, key: &str) -> Result<u64, (&'static str, String)> {
    params[key]
        .as_u64()
        .ok_or(("bad_request", format!("{key}: expected an integer")))
}

fn name(params: &Value, key: &str) -> Result<String, (&'static str, String)> {
    params[key]
        .as_str()
        .map(str::to_owned)
        .ok_or(("bad_request", format!("{key}: expected a string")))
}

impl FakeShell {
    pub fn with_contract(mut self, contract: u64, methods: &[&str]) -> Self {
        self.contract = contract;
        self.methods = methods.iter().map(|m| m.to_string()).collect();
        self
    }

    /// From now on, every file request waits `delay` before its reply.
    pub fn set_file_delay(&self, delay: Duration) {
        self.state.lock().unwrap().file_delay = delay;
    }

    pub fn answering_garbage(mut self, text: &'static str) -> Self {
        self.garbage = Some(text);
        self
    }

    /// A shell on the same files and bundle store (a relaunch), with a
    /// fresh log: its first document load applies the store's rules.
    pub fn relaunch(&self) -> Self {
        let (files, mut bundles) = {
            let s = self.state.lock().unwrap();
            (s.files.clone(), s.bundles.clone())
        };
        bundles.load_document();
        let shell = Self {
            state: Arc::default(),
            ..self.clone()
        };
        let mut s = shell.state.lock().unwrap();
        s.files = files;
        s.bundles = bundles;
        drop(s);
        shell
    }

    /// A document load in this shell (`location.reload()`).
    pub fn load_document(&self) {
        self.state.lock().unwrap().bundles.load_document();
    }

    pub fn bundles(&self) -> Bundles {
        self.state.lock().unwrap().bundles.clone()
    }

    /// The same for a shell and its relaunches.
    pub fn files_key(&self) -> Arc<()> {
        self.profile.clone()
    }

    pub fn log(&self) -> Vec<String> {
        self.state.lock().unwrap().log.clone()
    }

    fn note(&self, line: String) {
        self.state.lock().unwrap().log.push(line);
    }

    /// Apply a file or bundle request now (in arrival order).
    fn apply_now(&self, method: &str, p: &Value) -> Reply {
        let mut s = self.state.lock().unwrap();
        match method {
            "files.read" => {
                let n = name(p, "name")?;
                Ok(match s.files.get(&n) {
                    Some(data) => {
                        let mut data = data.clone();
                        if s.corrupt_reads
                            && let Some(last) = data.last_mut()
                        {
                            *last ^= 1;
                        }
                        json!({"data": STANDARD.encode(data)})
                    }
                    None => json!({"data": null}),
                })
            }
            "files.write" => {
                let n = name(p, "name")?;
                let data = bytes(p, "data")?;
                let offset = int(p, "offset")? as usize;
                let file = s.files.entry(n.clone()).or_default();
                if file.len() < offset + data.len() {
                    file.resize(offset + data.len(), 0);
                }
                file[offset..offset + data.len()].copy_from_slice(&data);
                let size = file.len();
                if p["sync"].as_bool() == Some(true) {
                    s.syncs.push(n);
                }
                Ok(json!({"size": size}))
            }
            "files.truncate" => {
                let n = name(p, "name")?;
                let size = int(p, "size")? as usize;
                s.files.entry(n).or_default().resize(size, 0);
                Ok(json!({}))
            }
            "files.sync" => {
                let n = name(p, "name")?;
                s.syncs.push(n);
                Ok(json!({}))
            }
            "files.rename" => {
                let (from, to) = (name(p, "from")?, name(p, "to")?);
                let Some(data) = s.files.remove(&from) else {
                    return err("not_found", from);
                };
                s.files.insert(to.clone(), data);
                s.syncs.push(to);
                Ok(json!({}))
            }
            "files.remove" => {
                let n = name(p, "name")?;
                s.files.remove(&n);
                Ok(json!({}))
            }
            "files.list" => Ok(json!({"names": s.files.keys().collect::<Vec<_>>()})),
            "bundle.begin" => {
                let v = name(p, "version")?;
                let b = &mut s.bundles;
                if v == "embedded" || b.versions.get(&v).is_some_and(|x| x.sealed) {
                    return err("invalid_state", "sealed or embedded");
                }
                b.versions.insert(v, StoredBundle::default());
                Ok(json!({}))
            }
            "bundle.write" => {
                let v = name(p, "version")?;
                let path = name(p, "path")?;
                if !eidola_app_core::updater::mobile::row::is_bundle_path(&path) {
                    return err("bad_request", path);
                }
                let offset = int(p, "offset")? as usize;
                let data = bytes(p, "data")?;
                let Some(version) = s.bundles.versions.get_mut(&v).filter(|x| !x.sealed) else {
                    return err("invalid_state", "not staging");
                };
                let file = version.files.entry(path).or_default();
                if file.len() < offset + data.len() {
                    file.resize(offset + data.len(), 0);
                }
                file[offset..offset + data.len()].copy_from_slice(&data);
                Ok(json!({}))
            }
            "bundle.seal" => {
                let v = name(p, "version")?;
                let Some(version) = s.bundles.versions.get_mut(&v).filter(|x| !x.sealed) else {
                    return err("invalid_state", "not staging");
                };
                let hashes: Vec<(String, String)> = version
                    .files
                    .iter()
                    .map(|(path, data)| (path.clone(), hex::encode(sha2::Sha256::digest(data))))
                    .collect();
                let digest = eidola_app_core::updater::mobile::row::bundle_digest(
                    hashes.iter().map(|(p, h)| (p.as_str(), h.as_str())),
                );
                version.sealed = true;
                version.digest = Some(digest.clone());
                Ok(json!({
                    "digest": digest,
                    "files": version.files.len(),
                    "bytes": version.files.values().map(Vec::len).sum::<usize>(),
                }))
            }
            "bundle.activate" => {
                let v = name(p, "version")?;
                let b = &mut s.bundles;
                if v == "embedded" {
                    b.active = v.clone();
                    b.previous = v;
                    b.confirmed = true;
                } else {
                    match b.versions.get(&v) {
                        None => return err("not_found", v),
                        Some(x) if !x.sealed => return err("invalid_state", "not sealed"),
                        Some(_) => {}
                    }
                    if b.confirmed {
                        b.previous = b.active.clone();
                    }
                    b.active = v;
                    b.confirmed = false;
                    b.tried = false;
                }
                Ok(b.state())
            }
            "bundle.confirm" => {
                s.confirms += 1;
                let b = &mut s.bundles;
                if b.serving == b.active && !b.confirmed {
                    b.confirmed = true;
                    b.serving_confirmed = true;
                    b.previous = "embedded".into();
                }
                Ok(b.state())
            }
            "bundle.remove" => {
                let v = name(p, "version")?;
                let b = &mut s.bundles;
                if v == "embedded" || v == b.active || v == b.previous || v == b.serving {
                    return err("invalid_state", "in use");
                }
                if b.versions.remove(&v).is_none() {
                    return err("not_found", v);
                }
                Ok(json!({}))
            }
            "bundle.state" => Ok(s.bundles.state()),
            "debug.syncLog" => Ok(json!({"syncs": std::mem::take(&mut s.syncs)})),
            _ => err("unknown_method", method),
        }
    }

    fn tcp(&self, method: &str, p: &Value) -> BoxFuture<'static, Reply> {
        let state = self.state.clone();
        let id = match int(p, "socket") {
            Ok(id) => id,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        match method {
            "tcp.connect" => {
                let host = p["host"].as_str().unwrap_or_default().to_owned();
                let port = p["port"].as_u64().unwrap_or_default() as u16;
                state.lock().unwrap().pending_connects.insert(id, ());
                Box::pin(async move {
                    let stream = TcpStream::connect((host.as_str(), port)).await;
                    let mut s = state.lock().unwrap();
                    if s.pending_connects.remove(&id).is_none() {
                        return err("closed", "closed while connecting");
                    }
                    let stream = stream.map_err(|e| ("connect_failed", e.to_string()))?;
                    stream.set_nodelay(true).ok();
                    let (read, mut write) = stream.into_split();
                    let (tx, mut outgoing) = mpsc::unbounded_channel::<Outgoing>();
                    RT.spawn(async move {
                        while let Some(next) = outgoing.recv().await {
                            match next {
                                Outgoing::Data(data, done) => {
                                    let r = write.write_all(&data).await;
                                    let _ = done.send(
                                        r.map(|()| json!({})).map_err(|e| ("io", e.to_string())),
                                    );
                                }
                                Outgoing::Shutdown(done) => {
                                    let r = write.shutdown().await;
                                    let _ = done.send(
                                        r.map(|()| json!({})).map_err(|e| ("io", e.to_string())),
                                    );
                                }
                            }
                        }
                    });
                    s.sockets.insert(
                        id,
                        Sock {
                            tx,
                            rx: Arc::new(tokio::sync::Mutex::new(read)),
                        },
                    );
                    Ok(json!({}))
                })
            }
            "tcp.send" | "tcp.shutdown" => {
                let (done, reply) = oneshot::channel();
                let message = if method == "tcp.send" {
                    match bytes(p, "data") {
                        Ok(data) => Outgoing::Data(data, done),
                        Err(e) => return Box::pin(async move { Err(e) }),
                    }
                } else {
                    Outgoing::Shutdown(done)
                };
                let sent = state
                    .lock()
                    .unwrap()
                    .sockets
                    .get(&id)
                    .map(|s| s.tx.send(message).is_ok());
                Box::pin(async move {
                    match sent {
                        Some(true) => reply.await.unwrap_or(err("closed", "closed")),
                        _ => err("not_found", "no such socket"),
                    }
                })
            }
            "tcp.receive" => {
                let max = p["max"].as_u64().unwrap_or(65536) as usize;
                let rx = {
                    let mut s = state.lock().unwrap();
                    let rx = s.sockets.get(&id).map(|s| s.rx.clone());
                    if rx.is_some() {
                        s.receives_in_flight += 1;
                        s.most_receives_in_flight =
                            s.most_receives_in_flight.max(s.receives_in_flight);
                    }
                    rx
                };
                Box::pin(async move {
                    let Some(rx) = rx else {
                        return err("not_found", "no such socket");
                    };
                    let mut buf = vec![0u8; max];
                    let n = rx.lock().await.read(&mut buf).await;
                    let mut s = state.lock().unwrap();
                    s.receives_in_flight -= 1;
                    match n {
                        Ok(0) => Ok(json!({"eof": true})),
                        Ok(n) => {
                            *s.received.entry(id).or_default() += n;
                            Ok(json!({"data": STANDARD.encode(&buf[..n])}))
                        }
                        Err(e) => err("io", e.to_string()),
                    }
                })
            }
            "tcp.close" => {
                let mut s = state.lock().unwrap();
                s.sockets.remove(&id);
                s.pending_connects.remove(&id);
                Box::pin(async { Ok(json!({})) })
            }
            _ => Box::pin(async { err("unknown_method", "") }),
        }
    }
}

impl Transport for FakeShell {
    fn call(&self, _id: u64, request: String) -> BoxFuture<'static, Result<String, String>> {
        let parsed: Value = serde_json::from_str(&request).unwrap_or(Value::Null);
        let id = parsed["id"].clone();
        let method = parsed["method"].as_str().unwrap_or_default().to_owned();
        let p = parsed["params"].clone();
        let target = p["name"]
            .as_str()
            .or(p["to"].as_str())
            .or(p["version"].as_str())
            .map(str::to_owned)
            .or_else(|| p["socket"].as_u64().map(|s| format!("socket {s}")))
            .unwrap_or_default();
        let label = format!("{method} {target}").trim_end().to_owned();
        self.note(format!("-> {label}"));
        let reply: BoxFuture<'static, Reply> = if let Some(text) = self.garbage {
            return Box::pin(async move { Ok(text.to_owned()) });
        } else if method == "shell.hello" {
            let hello = json!({
                "contract": self.contract, "platform": "fake", "shell": "test",
                "methods": self.methods,
            });
            Box::pin(async move { Ok(hello) })
        } else if method.starts_with("tcp.") {
            self.tcp(&method, &p)
        } else {
            let reply = self.apply_now(&method, &p);
            let delay = self.state.lock().unwrap().file_delay;
            Box::pin(async move {
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                reply
            })
        };
        let (tx, rx) = oneshot::channel();
        let state = self.state.clone();
        RT.spawn(async move {
            let reply = reply.await;
            let text = match reply {
                Ok(result) => json!({"id": id, "ok": true, "result": result}),
                Err((code, message)) => {
                    json!({"id": id, "ok": false, "error": {"code": code, "message": message}})
                }
            }
            .to_string();
            state.lock().unwrap().log.push(format!("<- {label}"));
            let _ = tx.send(text);
        });
        Box::pin(async move { rx.await.map_err(|_| "the fake shell dropped it".to_owned()) })
    }
}
