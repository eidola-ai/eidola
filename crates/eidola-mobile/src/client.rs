//! A typed client for every call of the shell contract, over a [`Transport`]
//! that moves request and reply texts: the webview's bridge in the app
//! (`crate::web`), a shell written in Rust in the native tests.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use serde::de::DeserializeOwned;
pub use tinfoil_verifier::pipe::{BoxFuture, MaybeSend};

use crate::protocol::{
    self, B64Ref, BundleState, Empty, FileData, Hello, Names, Received, Sealed, ShellError,
    SyncLog, Written,
};

/// Moves one request's text to the shell and its reply's text back.
///
/// `call` posts the request **before it returns**: the returned future only
/// waits for the reply, so dropping it does not withdraw the request (a
/// socket closed on drop relies on this). An `Err` is the bridge's own
/// failure, with a message for logs.
pub trait Transport: Send + Sync + 'static {
    fn call(&self, id: u64, request: String) -> BoxFuture<'static, Result<String, String>>;
}

/// One shell call, as the client saw it: for showing the order in which
/// requests left and replies came back (the self-test's barrier check).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceEntry {
    /// The request's id; a reply shares its request's.
    pub id: u64,
    pub method: &'static str,
    /// The file, bundle version or socket the call names, if any.
    pub target: String,
    pub event: TraceEvent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceEvent {
    /// Posted to the shell.
    Sent,
    /// The reply came back: whether it was `ok`.
    Replied { ok: bool },
}

/// What a call names, for the trace.
#[derive(Clone, Copy)]
pub enum Target<'a> {
    None,
    Name(&'a str),
    Socket(u64),
}

impl Target<'_> {
    fn render(self) -> String {
        match self {
            Target::None => String::new(),
            Target::Name(name) => name.to_owned(),
            Target::Socket(id) => format!("socket {id}"),
        }
    }
}

struct Inner<T> {
    transport: T,
    next_id: AtomicU64,
    next_socket: AtomicU64,
    tracing: AtomicBool,
    trace: Mutex<Vec<TraceEntry>>,
}

/// The shell, as typed calls. Cheap to clone; every clone shares one id
/// sequence.
pub struct ShellClient<T> {
    inner: Arc<Inner<T>>,
}

impl<T> Clone for ShellClient<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

#[derive(Serialize)]
struct NoParams {}

#[derive(Serialize)]
struct Name<'a> {
    name: &'a str,
}

#[derive(Serialize)]
struct Version<'a> {
    version: &'a str,
}

#[derive(Serialize)]
struct Socket {
    socket: u64,
}

impl<T: Transport> ShellClient<T> {
    pub fn new(transport: T) -> Self {
        Self {
            inner: Arc::new(Inner {
                transport,
                next_id: AtomicU64::new(1),
                next_socket: AtomicU64::new(1),
                tracing: AtomicBool::new(false),
                trace: Mutex::new(Vec::new()),
            }),
        }
    }

    /// Post `method` now; the future resolves to its typed result.
    pub fn call<P: Serialize, R: DeserializeOwned + MaybeSend + 'static>(
        &self,
        method: &'static str,
        target: Target<'_>,
        params: &P,
    ) -> BoxFuture<'static, Result<R, ShellError>> {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let traced = self.inner.tracing.load(Ordering::Relaxed);
        let target = if traced {
            target.render()
        } else {
            String::new()
        };
        let posted = protocol::encode(id, method, params).map(|text| {
            if traced {
                self.record(id, method, &target, TraceEvent::Sent);
            }
            self.inner.transport.call(id, text)
        });
        let this = traced.then(|| self.clone());
        Box::pin(async move {
            let reply = match posted {
                Ok(reply) => reply.await,
                Err(message) => Err(message),
            };
            let result = match reply {
                Ok(text) => protocol::decode(method, &text),
                Err(message) => Err(ShellError::Bridge { method, message }),
            };
            if let Some(this) = this {
                let ok = result.is_ok();
                this.record(id, method, &target, TraceEvent::Replied { ok });
            }
            result
        })
    }

    fn record(&self, id: u64, method: &'static str, target: &str, event: TraceEvent) {
        if let Ok(mut trace) = self.inner.trace.lock() {
            trace.push(TraceEntry {
                id,
                method,
                target: target.to_owned(),
                event,
            });
        }
    }

    /// Start recording every call (clearing what was recorded).
    pub fn start_trace(&self) {
        if let Ok(mut trace) = self.inner.trace.lock() {
            trace.clear();
        }
        self.inner.tracing.store(true, Ordering::Relaxed);
    }

    /// Stop recording, and hand back what was recorded.
    pub fn take_trace(&self) -> Vec<TraceEntry> {
        self.inner.tracing.store(false, Ordering::Relaxed);
        self.inner
            .trace
            .lock()
            .map(|mut trace| std::mem::take(&mut *trace))
            .unwrap_or_default()
    }

    /// A socket id no open socket of this client has.
    pub(crate) fn next_socket(&self) -> u64 {
        self.inner.next_socket.fetch_add(1, Ordering::Relaxed)
    }

    // shell.*

    pub async fn hello(&self) -> Result<Hello, ShellError> {
        self.call("shell.hello", Target::None, &NoParams {}).await
    }

    /// Hand an absolute `https:` URL to the system browser.
    pub async fn open_external(&self, url: &str) -> Result<(), ShellError> {
        #[derive(Serialize)]
        struct Url<'a> {
            url: &'a str,
        }
        self.call::<_, Empty>("shell.openExternal", Target::None, &Url { url })
            .await
            .map(drop)
    }

    // tcp.*

    pub fn tcp_connect(
        &self,
        socket: u64,
        host: &str,
        port: u16,
    ) -> BoxFuture<'static, Result<(), ShellError>> {
        #[derive(Serialize)]
        struct Connect<'a> {
            socket: u64,
            host: &'a str,
            port: u16,
        }
        let call = self.call::<_, Empty>(
            "tcp.connect",
            Target::Socket(socket),
            &Connect { socket, host, port },
        );
        Box::pin(async move { call.await.map(drop) })
    }

    pub fn tcp_send(&self, socket: u64, data: &[u8]) -> BoxFuture<'static, Result<(), ShellError>> {
        #[derive(Serialize)]
        struct Payload<'a> {
            socket: u64,
            data: B64Ref<'a>,
        }
        let call = self.call::<_, Empty>(
            "tcp.send",
            Target::Socket(socket),
            &Payload {
                socket,
                data: B64Ref(data),
            },
        );
        Box::pin(async move { call.await.map(drop) })
    }

    /// At most `max` bytes, or `None` at the end of the stream.
    pub fn tcp_receive(
        &self,
        socket: u64,
        max: usize,
    ) -> BoxFuture<'static, Result<Option<Vec<u8>>, ShellError>> {
        #[derive(Serialize)]
        struct Receive {
            socket: u64,
            max: usize,
        }
        let call = self.call::<_, Received>(
            "tcp.receive",
            Target::Socket(socket),
            &Receive { socket, max },
        );
        Box::pin(async move {
            match call.await? {
                Received {
                    data: Some(data), ..
                } if !data.0.is_empty() => Ok(Some(data.0)),
                Received { eof: true, .. } => Ok(None),
                _ => Err(ShellError::Malformed {
                    method: "tcp.receive",
                    detail: "neither bytes nor the end of the stream".into(),
                }),
            }
        })
    }

    pub fn tcp_shutdown(&self, socket: u64) -> BoxFuture<'static, Result<(), ShellError>> {
        let call =
            self.call::<_, Empty>("tcp.shutdown", Target::Socket(socket), &Socket { socket });
        Box::pin(async move { call.await.map(drop) })
    }

    /// Posted at once; awaiting the future is optional.
    pub fn tcp_close(&self, socket: u64) -> BoxFuture<'static, Result<(), ShellError>> {
        let call = self.call::<_, Empty>("tcp.close", Target::Socket(socket), &Socket { socket });
        Box::pin(async move { call.await.map(drop) })
    }

    // files.*

    /// The whole file, or `None` when it is missing.
    pub async fn files_read(&self, name: &str) -> Result<Option<Vec<u8>>, ShellError> {
        let read: FileData = self
            .call("files.read", Target::Name(name), &Name { name })
            .await?;
        Ok(read.data.map(|data| data.0))
    }

    /// Write `data` at `offset` (at most [`protocol::MAX_REQUEST_BYTES`]);
    /// the file's length after the write.
    pub async fn files_write(
        &self,
        name: &str,
        offset: u64,
        data: &[u8],
        sync: bool,
    ) -> Result<u64, ShellError> {
        #[derive(Serialize)]
        struct Write<'a> {
            name: &'a str,
            offset: u64,
            data: B64Ref<'a>,
            #[serde(skip_serializing_if = "std::ops::Not::not")]
            sync: bool,
        }
        let written: Written = self
            .call(
                "files.write",
                Target::Name(name),
                &Write {
                    name,
                    offset,
                    data: B64Ref(data),
                    sync,
                },
            )
            .await?;
        Ok(written.size)
    }

    pub async fn files_truncate(&self, name: &str, size: u64) -> Result<(), ShellError> {
        #[derive(Serialize)]
        struct Truncate<'a> {
            name: &'a str,
            size: u64,
        }
        self.call::<_, Empty>(
            "files.truncate",
            Target::Name(name),
            &Truncate { name, size },
        )
        .await
        .map(drop)
    }

    pub async fn files_sync(&self, name: &str) -> Result<(), ShellError> {
        self.call::<_, Empty>("files.sync", Target::Name(name), &Name { name })
            .await
            .map(drop)
    }

    pub async fn files_rename(&self, from: &str, to: &str) -> Result<(), ShellError> {
        #[derive(Serialize)]
        struct Rename<'a> {
            from: &'a str,
            to: &'a str,
        }
        self.call::<_, Empty>("files.rename", Target::Name(to), &Rename { from, to })
            .await
            .map(drop)
    }

    pub async fn files_remove(&self, name: &str) -> Result<(), ShellError> {
        self.call::<_, Empty>("files.remove", Target::Name(name), &Name { name })
            .await
            .map(drop)
    }

    pub async fn files_list(&self) -> Result<Vec<String>, ShellError> {
        let names: Names = self.call("files.list", Target::None, &NoParams {}).await?;
        Ok(names.names)
    }

    // bundle.*

    pub async fn bundle_begin(&self, version: &str) -> Result<(), ShellError> {
        self.call::<_, Empty>("bundle.begin", Target::Name(version), &Version { version })
            .await
            .map(drop)
    }

    /// Write into a staging version's file (at most
    /// [`protocol::MAX_REQUEST_BYTES`] per call).
    pub async fn bundle_write(
        &self,
        version: &str,
        path: &str,
        offset: u64,
        data: &[u8],
    ) -> Result<(), ShellError> {
        #[derive(Serialize)]
        struct Write<'a> {
            version: &'a str,
            path: &'a str,
            offset: u64,
            data: B64Ref<'a>,
        }
        self.call::<_, Empty>(
            "bundle.write",
            Target::Name(version),
            &Write {
                version,
                path,
                offset,
                data: B64Ref(data),
            },
        )
        .await
        .map(drop)
    }

    pub async fn bundle_seal(&self, version: &str) -> Result<Sealed, ShellError> {
        self.call("bundle.seal", Target::Name(version), &Version { version })
            .await
    }

    pub async fn bundle_activate(&self, version: &str) -> Result<BundleState, ShellError> {
        self.call(
            "bundle.activate",
            Target::Name(version),
            &Version { version },
        )
        .await
    }

    /// Tell the shell the running bundle booted.
    pub async fn bundle_confirm(&self) -> Result<BundleState, ShellError> {
        self.call("bundle.confirm", Target::None, &NoParams {})
            .await
    }

    pub async fn bundle_remove(&self, version: &str) -> Result<(), ShellError> {
        self.call::<_, Empty>("bundle.remove", Target::Name(version), &Version { version })
            .await
            .map(drop)
    }

    pub async fn bundle_state(&self) -> Result<BundleState, ShellError> {
        self.call("bundle.state", Target::None, &NoParams {}).await
    }

    /// A debug shell's flush record since the last call (outside the
    /// contract: a release shell answers `unknown_method`).
    pub async fn debug_sync_log(&self) -> Result<Vec<String>, ShellError> {
        let log: SyncLog = self
            .call("debug.syncLog", Target::None, &NoParams {})
            .await?;
        Ok(log.syncs)
    }
}
