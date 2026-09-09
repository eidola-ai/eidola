//! The local inference proxy's listener — the second door this process opens.
//!
//! The local database has exactly one writer and this process is holding it, so
//! anything answering for the profile has to live here. That is the whole
//! reason [`crate::ipc`]'s control socket is in this crate, and it is the same
//! reason this is: a proxied request spends from the wallet, reads the backend
//! registry, and writes to the Record, none of which a second process can do.
//!
//! **The disciplines are the control socket's, deliberately.** Binding,
//! admission bounds, a shutdown that ends established work, and a teardown
//! taxonomy honest about what it stops — every one of those was learned once on
//! [`crate::ipc`] and every one applies here, because this door can start
//! *billed* work exactly as that one can. What differs is only what the two
//! doors are for, so the shapes are shared rather than re-derived:
//!
//! | | Control socket | This |
//! |---|---|---|
//! | Transport | Unix socket in the data dir | TCP, address and port the reader chose |
//! | Who may talk | tier-0 peer credentials (same uid) | an API key the reader generated |
//! | Lifetime | bound at launch, closed only by a full shutdown | started and stopped on demand *and* closed only by a full shutdown |
//!
//! ## Who may talk to it, and the honest frame about that
//!
//! There is no peer-credential check here and there cannot be one: TCP has no
//! `SO_PEERCRED`, and the whole point of a bindable address is that the peer
//! may not be on this machine. Authentication is therefore the API key alone
//! (`eidola_app_core::proxy::http`), and **a proxy with no live keys refuses
//! everything** rather than running open.
//!
//! The honest frame: bound to loopback with a key, this exposes to a local
//! process what that process could already get by driving the app — no more,
//! and with an audit trail the app itself leaves. Bound to a routable address
//! it is a plaintext inference endpoint on the network, which is a genuinely
//! different thing, and the surface that sets the address says so. TLS is a
//! later feature; until it exists, loopback is the shape this is safe in.
//!
//! ## Lifetime
//!
//! Started when the reader enables it and whenever its binding moves; stopped
//! when they disable it. **⌘Q's retire does not stop it** — the app outliving
//! its windows is the point, and a tool pointed at this proxy must not lose its
//! endpoint because somebody closed a conversation. Only the full shutdown
//! closes the door, from the same hook that closes the control socket and
//! drains the engines, and for the same reason: everything after it is
//! teardown, and a caller still being served can start a billed turn while it
//! runs.

use std::io;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use eidola_app_core::AppCore;
use eidola_app_core::error::AppError;
use eidola_app_core::ipc::Shutdown;
use eidola_app_core::proxy::{ProxySettings, parse_bind_address};

/// How many consecutive `accept` failures the loop tolerates before giving up.
///
/// The control socket's reasoning, unchanged: some accept errors are transient
/// and self-clearing, some are permanent, and a listener that can only fail
/// must not spin.
const MAX_CONSECUTIVE_ACCEPT_FAILURES: u32 = 16;

/// How long the accept loop waits after a failure before trying again.
const ACCEPT_RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(100);

/// How many connections this proxy serves at once.
///
/// Higher than the control socket's eight, because HTTP keep-alive means a
/// *connection* here is a client that may be idle rather than a request in
/// flight: an editor extension, a terminal agent and a notebook kernel each
/// hold one open between asks. Still a stated number rather than "as many as
/// the peer opens", because what one of those connections can start is billed.
///
/// A peer past the cap is dropped rather than told why: a `503` would need a
/// response written on a connection the process has just decided not to serve,
/// and every correct client reconnects.
const MAX_CONNECTIONS: usize = 32;

/// The connections being served, and whether the door has closed.
///
/// **The flag and the set live under one lock because the race is between
/// them** — the control socket's `Serving`, and for the identical reason:
/// aborting the accept loop only *schedules* its cancellation, so an accept
/// already in hand runs on through straight-line code and reaches the insert.
/// Sweeping the set and then being handed a connection is a peer served by a
/// door that has closed, with billed work one request away. Under one flag that
/// insertion is unrepresentable.
///
/// A `std` mutex, since it is held only across the synchronous half of that
/// decision and never across an `await` — [`ProxyServer::close`] runs on the
/// quit path, off any runtime, and has to be able to take it.
#[derive(Default)]
struct Serving {
    closed: bool,
    connections: tokio::task::JoinSet<()>,
    /// Cloned into every connection this set admits, so the shut below reaches
    /// the ones already inside.
    shutdown: Shutdown,
}

impl Serving {
    fn admit(&mut self, serve: impl std::future::Future<Output = ()> + Send + 'static) -> Admitted {
        if self.closed {
            return Admitted::DoorShut;
        }
        // Reaped **before** the count: a stale tally would turn callers away on
        // behalf of connections that hung up long ago. Never blocks the loop.
        while self.connections.try_join_next().is_some() {}
        if self.connections.len() >= MAX_CONNECTIONS {
            return Admitted::AtCapacity;
        }
        self.connections.spawn(serve);
        Admitted::Yes
    }

    /// Shut the door, and end everyone already through it.
    ///
    /// Three layers, three moments: `closed` refuses a connection arriving from
    /// here on; the shutdown latch stops an *already admitted* one from
    /// dispatching anything more; the abort ends the tasks themselves. Latched
    /// **before** the abort, so nothing can read it as still open in the window
    /// between the two.
    fn shut(&mut self) {
        self.closed = true;
        self.shutdown.latch();
        self.connections.abort_all();
    }
}

/// What the door said to one connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Admitted {
    Yes,
    /// The listener closed while this connection was in hand.
    DoorShut,
    /// [`MAX_CONNECTIONS`] are already being served.
    AtCapacity,
}

type Connections = Arc<Mutex<Serving>>;

/// Why the accept loop stopped, when it stopped on its own.
///
/// **A listener that has given up is not a listener.** The loop returns after
/// [`MAX_CONSECUTIVE_ACCEPT_FAILURES`] and the socket goes with it, but the
/// handle would go on holding a `ProxyServer` — so `address()` kept naming a
/// port nothing was accepting on, the pane kept saying it was listening, and
/// every later reconciliation saw the wanted address already bound and refused
/// to start it again. The state is therefore shared rather than local to the
/// task: what the loop learns, the handle answers with.
type AcceptEnded = Arc<Mutex<Option<String>>>;

/// A bound listener, held for as long as the proxy is running.
pub struct ProxyServer {
    address: SocketAddr,
    accepting: tokio::task::JoinHandle<()>,
    connections: Connections,
    ended: AcceptEnded,
}

impl ProxyServer {
    /// The address this listener bound, or `None` once it has stopped
    /// accepting — a socket that admits nobody is not somewhere to point a
    /// tool.
    pub fn address(&self) -> Option<SocketAddr> {
        self.accept_failure().is_none().then_some(self.address)
    }

    /// The address it bound, whatever has become of it. What a diagnostic
    /// names; never what a reader is told to point at.
    pub fn bound_address(&self) -> SocketAddr {
        self.address
    }

    /// Why accepting stopped, if it has.
    pub fn accept_failure(&self) -> Option<String> {
        self.ended.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Test-only: put the listener into the state its accept loop reaches after
    /// [`MAX_CONSECUTIVE_ACCEPT_FAILURES`].
    ///
    /// That state is real and unreachable from a test — it takes an operating
    /// system that refuses sixteen accepts in a row — so the seam exists to
    /// exercise what the *handle* then answers, which is the half this crate
    /// owns.
    #[doc(hidden)]
    pub fn fail_accepting_for_test(&self, reason: &str) {
        *self.ended.lock().unwrap_or_else(|e| e.into_inner()) = Some(reason.to_string());
    }

    /// Stop serving.
    ///
    /// Ends the accept loop and the connections it accepted. Ending a
    /// connection is the same event a peer that hung up produces, so it costs
    /// the frames and never the work — a turn already upstream still lands and
    /// still settles its credential.
    pub fn close(&self) {
        self.accepting.abort();
        self.connections
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .shut();
    }
}

/// Bind and serve the proxy for one open core.
pub fn serve(core: &Arc<AppCore>, settings: &ProxySettings) -> Result<ProxyServer, AppError> {
    let ip = parse_bind_address(&settings.bind_address)?;
    let address = SocketAddr::new(ip, settings.bind_port);
    let listener = bind_listener(address).map_err(|e| AppError::Config {
        message: format!("could not listen on {address}: {e}"),
    })?;
    // The bound address rather than the requested one — they differ if a port
    // of 0 is ever allowed, and a surface that tells a reader where to point
    // their tool must state what happened, not what was asked for.
    let address = listener.local_addr().unwrap_or(address);
    let connections: Connections = Default::default();
    let shutdown = connections
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .shutdown
        .clone();
    let ended: AcceptEnded = Default::default();
    let accepting = core.runtime().spawn(accept_loop(
        Arc::clone(core),
        listener,
        Arc::clone(&connections),
        shutdown,
        Arc::clone(&ended),
    ));
    Ok(ProxyServer {
        address,
        accepting,
        connections,
        ended,
    })
}

fn bind_listener(address: SocketAddr) -> io::Result<std::net::TcpListener> {
    let listener = std::net::TcpListener::bind(address)?;
    listener.set_nonblocking(true)?;
    Ok(listener)
}

async fn accept_loop(
    core: Arc<AppCore>,
    listener: std::net::TcpListener,
    connections: Connections,
    shutdown: Shutdown,
    ended: AcceptEnded,
) {
    // Every way out of this loop that is not a deliberate close records why,
    // so the handle can stop claiming to be listening.
    let give_up = |reason: String| {
        eprintln!("eidola-gui: the local inference proxy stopped accepting: {reason}");
        *ended.lock().unwrap_or_else(|e| e.into_inner()) = Some(reason);
    };
    let listener = match tokio::net::TcpListener::from_std(listener) {
        Ok(listener) => listener,
        Err(e) => {
            give_up(e.to_string());
            return;
        }
    };
    let mut failures = 0u32;
    loop {
        let stream = match listener.accept().await {
            Ok((stream, _peer)) => {
                failures = 0;
                stream
            }
            Err(e) => {
                failures += 1;
                if failures >= MAX_CONSECUTIVE_ACCEPT_FAILURES {
                    give_up(e.to_string());
                    return;
                }
                tokio::time::sleep(ACCEPT_RETRY_DELAY).await;
                continue;
            }
        };
        // Answers are small and latency matters more than packet count on a
        // stream of SSE chunks a person is watching arrive.
        let _ = stream.set_nodelay(true);

        let core = Arc::clone(&core);
        let shutdown = shutdown.clone();
        let admitted = connections
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .admit(async move {
                eidola_app_core::proxy::http::serve_connection(core, stream, shutdown).await;
            });
        match admitted {
            Admitted::Yes => {}
            Admitted::DoorShut => {
                // The listener closed with this connection in hand — dropping
                // the future drops the stream, so the peer sees the same
                // hang-up every other caller just got. Returning rather than
                // looping: the door is shut and the abort we are racing is on
                // its way regardless.
                return;
            }
            Admitted::AtCapacity => {
                eprintln!(
                    "eidola-gui: refused a proxy connection — already serving {MAX_CONNECTIONS}"
                );
            }
        }
    }
}

/// The process's handle on the proxy, shared by whoever starts it and whoever
/// has to close it.
///
/// It exists because those are two different owners: the settings store starts
/// and stops the listener as the reader's configuration moves, and the full
/// shutdown hook closes it as the first step of teardown — the same closure
/// that closes the control socket, so the order between the two doors and the
/// engine drain stays the order of lines in one body rather than the sequence
/// of separately-registered hooks.
#[derive(Clone, Default)]
pub struct ProxyHandle {
    server: Arc<Mutex<Option<ProxyServer>>>,
    /// How many sockets this handle has bound.
    ///
    /// **A restart is not observable from outside without it.** Whether a
    /// reconcile left a correct listener alone or closed and rebound it on the
    /// same address is invisible in the address it reports — and the *cost* of
    /// getting that wrong is a race (the old socket may or may not have been
    /// released), which is exactly the kind of thing a test must not depend on.
    /// One counter makes "nothing was started" a fact rather than a hope.
    binds: Arc<std::sync::atomic::AtomicU64>,
}

impl ProxyHandle {
    /// Start (or restart) the listener on `settings`' binding.
    ///
    /// **Restart, not reconfigure**: a bound socket's address cannot move, so
    /// the running listener is closed before the new one binds — which also
    /// means an address the OS refuses leaves the proxy *stopped* rather than
    /// still answering on the old one. That is the honest outcome: a reader who
    /// changed the address must not be told it moved when it did not.
    pub fn start(
        &self,
        core: &Arc<AppCore>,
        settings: &ProxySettings,
    ) -> Result<SocketAddr, AppError> {
        let mut held = self.server.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(running) = held.take() {
            running.close();
        }
        let server = serve(core, settings)?;
        // The bound address — a listener a frame old has not given up, and
        // `bound_address` is the one that always answers.
        let address = server.bound_address();
        *held = Some(server);
        self.binds
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(address)
    }

    /// Stop the listener, if one is running. Idempotent.
    pub fn stop(&self) {
        let mut held = self.server.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(running) = held.take() {
            running.close();
        }
    }

    /// The address the listener is bound to, or `None` when it is not running.
    ///
    /// **Derived from the listener itself, never from the stored setting** —
    /// "the reader asked for this" and "this is where a tool should point" are
    /// different facts, and a bind that failed makes them differ.
    pub fn address(&self) -> Option<SocketAddr> {
        self.server
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .and_then(ProxyServer::address)
    }

    /// Why the listener stopped accepting, if it did.
    ///
    /// **Read at the reconcile, which is also where it is acted on**: a
    /// listener that gave up answers no address, so the next reconciliation
    /// sees nothing bound where the settings want something and starts it
    /// again. Nothing here pushes — the accept loop runs on the core's runtime
    /// and the bus is app-core's to emit on — so the pane learns at its next
    /// notify. Sixteen consecutive refused accepts is an operating system that
    /// has stopped answering, so the honest cost is that a reader may see the
    /// stale line until something else moves.
    pub fn accept_failure(&self) -> Option<String> {
        self.server
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .and_then(ProxyServer::accept_failure)
    }

    /// Whether the listener is running.
    pub fn is_running(&self) -> bool {
        self.address().is_some()
    }

    /// Test-only: how many sockets this handle has bound.
    #[doc(hidden)]
    pub fn binds_for_test(&self) -> u64 {
        self.binds.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Test-only: see [`ProxyServer::fail_accepting_for_test`].
    #[doc(hidden)]
    pub fn fail_accepting_for_test(&self, reason: &str) {
        if let Some(server) = self
            .server
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            server.fail_accepting_for_test(reason);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_connection_arriving_after_the_sweep_is_refused() {
        // The interleaving this exists for cannot be scheduled from a test: the
        // accept loop's body between two `accept` calls is straight-line code,
        // so an abort landing inside it takes effect only afterwards and the
        // insert happens regardless. What *is* testable is the decision that
        // makes either outcome harmless — a set that has been swept refuses the
        // connection instead of taking it on.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime");
        let _guard = runtime.enter();

        let mut serving = Serving::default();
        assert_eq!(
            serving.admit(std::future::pending()),
            Admitted::Yes,
            "an open door takes the connection"
        );

        serving.shut();
        assert_eq!(
            serving.admit(std::future::pending()),
            Admitted::DoorShut,
            "a connection accepted a moment before the close was served past it"
        );
    }

    #[test]
    fn shutting_reaches_the_connections_already_inside() {
        // The refusal above covers the door. This covers everyone already
        // through it: the latch they each hold is set, so the next request one
        // of them reads is refused rather than dispatched — which an abort
        // alone cannot promise, because it lands only at an await point.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime");
        let _guard = runtime.enter();

        let mut serving = Serving::default();
        let held = serving.shutdown.clone();
        assert!(!held.is_latched(), "a serving proxy is not shutting down");

        serving.shut();
        assert!(
            held.is_latched(),
            "a connection admitted before the close would keep dispatching billed work"
        );
    }

    #[test]
    fn a_client_that_opens_connections_without_end_is_bounded() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime");
        let _guard = runtime.enter();

        let mut serving = Serving::default();
        for i in 0..MAX_CONNECTIONS {
            assert_eq!(
                serving.admit(std::future::pending()),
                Admitted::Yes,
                "connection {i} is within the cap"
            );
        }
        assert_eq!(
            serving.admit(std::future::pending()),
            Admitted::AtCapacity,
            "what one of these can start is billed, so the aggregate is a stated number"
        );
    }

    #[test]
    fn a_stopped_handle_is_bound_to_nothing() {
        let handle = ProxyHandle::default();
        assert!(!handle.is_running());
        assert_eq!(handle.address(), None);
        // Idempotent: stopping what is not running is not an error.
        handle.stop();
        assert!(!handle.is_running());
    }
}
