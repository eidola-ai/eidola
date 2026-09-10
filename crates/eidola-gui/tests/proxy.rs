//! The inference proxy's listener, over a real TCP socket with a real client.
//!
//! The wire conversation itself is app-core's (`tests/proxy.rs`, over an
//! in-memory duplex). What only this tier can show is the part that is about
//! the *socket*: that a bound listener answers a real client on the address it
//! reports, that closing takes the door away for a client that was already
//! connected, and that a bind the OS refuses is reported rather than silently
//! leaving the reader thinking their tool has somewhere to point.
//!
//! Every test binds **port 0** and reads the address back off the listener,
//! which is why the whole suite can run in parallel with everything else: it
//! never claims a fixed port, and the reported address is what a client dials.
//! (Port 0 is deliberately *not* something the settings surface accepts — a
//! stored binding that changes on every restart is a config file that lies —
//! so these tests hand `serve` a `ProxySettings` value directly, which is the
//! seam that exists for exactly this.)

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;

use eidola_app_core::AppCore;
use eidola_app_core::proxy::{LocalExposure, ProxySettings};
use eidola_gui::proxy;

/// A real `AppCore` over tempdirs. Nothing here reaches the network.
fn core() -> (Arc<AppCore>, tempfile::TempDir) {
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let dir = tempfile::tempdir().expect("tempdir");
    let config_dir = dir.path().to_path_buf();
    let data_dir = dir.path().join("data");
    let core = AppCore::new(config_dir, data_dir).expect("open core");
    (Arc::new(core), dir)
}

/// Settings that bind an ephemeral loopback port.
fn ephemeral() -> ProxySettings {
    ProxySettings {
        enabled: true,
        bind_address: "127.0.0.1".into(),
        bind_port: 0,
        local_exposure: LocalExposure::Loaded,
        backends: Vec::new(),
        exposed_ids: Vec::new(),
        live_key_count: 0,
    }
}

/// One request over a real socket; the whole response as text.
fn ask(address: std::net::SocketAddr, request: &str) -> String {
    let mut stream = TcpStream::connect(address).expect("connect");
    stream.write_all(request.as_bytes()).expect("write");
    stream.flush().expect("flush");
    let mut text = String::new();
    stream.read_to_string(&mut text).expect("read");
    text
}

fn get(path: &str, key: Option<&str>) -> String {
    let auth = key
        .map(|k| format!("Authorization: Bearer {k}\r\n"))
        .unwrap_or_default();
    format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n{auth}\r\n")
}

#[test]
fn a_bound_proxy_answers_a_real_client_where_it_says_it_is() {
    let (core, _dir) = core();
    let key = core
        .runtime()
        .block_on(core.create_proxy_key("a tool".into()))
        .expect("mint")
        .key;

    let server = proxy::serve(&core, &ephemeral()).expect("bind");
    let address = server.address().expect("a listener that has not given up");
    assert!(
        address.ip().is_loopback() && address.port() != 0,
        "the address reported is the one actually bound: {address}"
    );

    // A key the reader generated gets in; anything else does not, and both
    // answers come over a real socket rather than a duplex.
    assert!(ask(address, &get("/v1/models", Some(&key))).starts_with("HTTP/1.1 200"));
    assert!(ask(address, &get("/v1/models", None)).starts_with("HTTP/1.1 401"));

    server.close();
}

#[test]
fn closing_takes_the_door_away_from_a_client_that_was_already_connected() {
    let (core, _dir) = core();
    let key = core
        .runtime()
        .block_on(core.create_proxy_key("a tool".into()))
        .expect("mint")
        .key;
    let server = proxy::serve(&core, &ephemeral()).expect("bind");
    let address = server.bound_address();

    // A client that is connected and has not yet asked for anything. Closing
    // means closed to everyone, not only to newcomers — the control socket's
    // rule, and it exists here for the same reason: this door starts billed
    // work, and everything after the close is teardown.
    //
    // **What this has teeth for is the outcome, and deliberately so.** Three
    // layers produce it and they compose rather than overlap: the `closed`
    // flag refuses an accept arriving from here on, the shutdown latch stops a
    // connection already inside from dispatching, and the abort ends the task.
    // Which of them got there first is a scheduling question no test can pin
    // without becoming a race, so each is pinned alone by its own unit test
    // (`proxy::tests::{a_connection_arriving_after_the_sweep_is_refused,
    // shutting_reaches_the_connections_already_inside}`) and this one asserts
    // the only thing that matters to the peer: it is not served.
    let mut established = TcpStream::connect(address).expect("connect");
    // Give the accept loop a moment to take the connection on, so the close
    // below is genuinely ending an established one rather than racing it.
    std::thread::sleep(std::time::Duration::from_millis(100));

    server.close();

    established
        .write_all(get("/v1/models", Some(&key)).as_bytes())
        .expect("the socket is still writable until the peer notices");
    let _ = established.flush();
    let mut text = String::new();
    let _ = established.read_to_string(&mut text);
    assert!(
        !text.contains("HTTP/1.1 200"),
        "a connection the close swept must not be served: {text:?}"
    );

    // And nothing new gets in either.
    assert!(
        TcpStream::connect(address)
            .and_then(|mut s| {
                s.write_all(get("/v1/models", Some(&key)).as_bytes())?;
                let mut t = String::new();
                s.read_to_string(&mut t)?;
                Ok(t)
            })
            .map(|t| !t.contains("HTTP/1.1 200"))
            .unwrap_or(true),
        "the listener is gone"
    );
}

#[test]
fn a_binding_the_system_refuses_is_reported_rather_than_pretended() {
    let (core, _dir) = core();

    // Hold the port, then ask the proxy for it.
    let squatter = std::net::TcpListener::bind("127.0.0.1:0").expect("squat");
    let taken = squatter.local_addr().expect("addr").port();
    let settings = ProxySettings {
        bind_port: taken,
        ..ephemeral()
    };
    let refused = proxy::serve(&core, &settings);
    assert!(
        refused.is_err(),
        "a port already held is a refusal, not a silent no-op"
    );
    let message = refused.err().expect("the error").to_string();
    assert!(
        message.contains(&taken.to_string()),
        "the refusal names the address the reader chose: {message}"
    );

    // An address this machine does not have is refused the same way — the
    // pane's `listen_error` is what carries either of these.
    let elsewhere = ProxySettings {
        bind_address: "203.0.113.1".into(),
        ..ephemeral()
    };
    assert!(proxy::serve(&core, &elsewhere).is_err());
}

/// REGRESSION: **a listener that has given up is not a listener.**
///
/// The accept loop returns after sixteen consecutive refused accepts and the
/// socket goes with it, but the handle went on holding the `ProxyServer` — so
/// `address()` kept naming a port nothing was accepting on, the pane kept
/// saying it was listening, and every later reconciliation saw the wanted
/// address already bound and refused to start it again. Terminal in the one
/// place a terminal state is worst: silently, on the door a reader pointed a
/// tool at.
///
/// The cure is that the handle answers with what the loop learned, which makes
/// the recovery fall out of the reconcile rather than needing a second
/// mechanism: no address means nothing is bound where the settings want
/// something, so the next reconcile starts it again.
#[test]
fn a_listener_that_gave_up_stops_claiming_an_address_and_can_be_started_again() {
    let (core, _dir) = core();
    let handle = proxy::ProxyHandle::default();
    let first = handle.start(&core, &ephemeral()).expect("start");
    assert_eq!(handle.address(), Some(first));
    assert!(handle.is_running());
    assert_eq!(handle.accept_failure(), None);

    handle.fail_accepting_for_test("too many open files");

    assert_eq!(
        handle.address(),
        None,
        "a socket that admits nobody is not somewhere to point a tool"
    );
    assert!(
        !handle.is_running(),
        "and the pane must not say it is running"
    );
    assert_eq!(
        handle.accept_failure().as_deref(),
        Some("too many open files"),
        "the reason is carried out, so the surface can say more than 'stopped'"
    );

    // Reconcilable again: nothing special-cases the terminal state — a
    // reconcile sees no address where the settings want one and starts it.
    let second = handle.start(&core, &ephemeral()).expect("restart");
    assert_eq!(handle.address(), Some(second));
    assert_eq!(
        handle.accept_failure(),
        None,
        "the fresh listener answers for itself"
    );
    handle.stop();
}

#[test]
fn a_restart_leaves_nothing_answering_on_the_address_it_left() {
    let (core, _dir) = core();
    let handle = proxy::ProxyHandle::default();
    assert!(!handle.is_running(), "nothing is bound until it is started");

    let first = handle.start(&core, &ephemeral()).expect("start");
    assert_eq!(handle.address(), Some(first));

    // **A restart, not a reconfigure**: a bound socket's address cannot move,
    // so the running listener is closed before the new one binds.
    let second = handle.start(&core, &ephemeral()).expect("restart");
    assert_ne!(first, second, "an ephemeral rebind lands somewhere new");
    assert_eq!(handle.address(), Some(second));
    assert!(
        TcpStream::connect(first)
            .and_then(|mut s| {
                s.write_all(get("/v1/models", None).as_bytes())?;
                let mut t = String::new();
                s.read_to_string(&mut t)?;
                Ok(t)
            })
            .map(|t| t.is_empty())
            .unwrap_or(true),
        "the address it left answers nothing"
    );

    handle.stop();
    assert!(!handle.is_running());
    // Idempotent: stopping what is not running is not an error.
    handle.stop();
    assert_eq!(handle.address(), None);
}

/// REGRESSION: **an address is compared as a value, never as text.**
///
/// `SocketAddr`'s own `Display` brackets an IPv6 host — `[::1]:11437` — while a
/// `host:port` join does not, so the reconcile's "is what is bound what the
/// settings describe" question answered *no* for every IPv6 listener, however
/// correct. Every refresh then restarted it; and since a restart closes before
/// it binds, an unrelated invalidation could leave the endpoint stopped, on the
/// door a reader had pointed a tool at.
///
/// `::1` is explicitly supported (`parse_bind_address` accepts it and
/// `is_loopback` calls it safe), so this is an ordinary configuration rather
/// than an exotic one.
#[test]
fn an_ipv6_listener_is_recognised_as_the_one_the_settings_describe() {
    // The two spellings of one address: what a naive join produces, and what
    // the socket actually reports.
    let listener = std::net::TcpListener::bind("[::1]:0").expect("an IPv6 loopback listener");
    let bound = listener.local_addr().expect("addr");
    let joined = format!("{}:{}", "::1", bound.port());
    assert_ne!(
        joined,
        bound.to_string(),
        "the string comparison this replaced could never match"
    );

    let ip = eidola_app_core::proxy::parse_bind_address("::1").expect("::1 parses");
    assert_eq!(
        std::net::SocketAddr::new(ip, bound.port()),
        bound,
        "compared as values, the settings and the socket are the same address"
    );
}

#[test]
fn a_proxy_bound_to_ipv6_loopback_answers_there() {
    let (core, _dir) = core();
    let key = core
        .runtime()
        .block_on(core.create_proxy_key("a tool".into()))
        .expect("mint")
        .key;

    let settings = ProxySettings {
        bind_address: "::1".into(),
        ..ephemeral()
    };
    let server = proxy::serve(&core, &settings).expect("bind ::1");
    let address = server.address().expect("a listener that has not given up");
    assert!(address.is_ipv6(), "bound where the reader asked: {address}");
    assert!(ask(address, &get("/v1/models", Some(&key))).starts_with("HTTP/1.1 200"));
    server.close();
}

/// **The full shutdown latches the door; a reconcile landing after it starts
/// nothing.**
///
/// `stop()` is reversible by design — the reader's switch turns the proxy off
/// and on again — so the shutdown needs a stronger word than the one the store
/// uses every day. A `ProxyStore` read or write already in flight when teardown
/// begins runs its continuation during the shutdown grace, calls
/// `reconcile_listener`, sees the settings still asking for a proxy, and binds:
/// the endpoint reopening after the doors were closed, ready to accept billed
/// work while the engines drain.
///
/// The counter is what makes "nothing was started" a fact rather than a hope —
/// a rebind on an ephemeral port would land somewhere new and prove nothing
/// about whether one happened at all.
#[test]
fn a_reconcile_landing_after_the_shutdown_starts_nothing() {
    let (core, _dir) = core();
    let handle = proxy::ProxyHandle::default();
    let address = handle.start(&core, &ephemeral()).expect("start");
    assert_eq!(handle.address(), Some(address));
    let bound = handle.binds_for_test();

    handle.retire();
    assert_eq!(handle.address(), None, "the door is closed");

    // What a store operation completing inside the grace would do.
    let refused = handle
        .start(&core, &ephemeral())
        .expect_err("a retired handle starts nothing");
    assert!(
        refused.to_string().contains("shutdown"),
        "and says why: {refused}"
    );
    assert_eq!(
        handle.binds_for_test(),
        bound,
        "no socket was bound after the door closed"
    );
    assert!(!handle.is_running());

    // The latch does not clear: a second reconcile is refused too.
    assert!(handle.start(&core, &ephemeral()).is_err());
    assert_eq!(handle.binds_for_test(), bound);
}
