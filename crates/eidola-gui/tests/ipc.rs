//! The control socket, over a real Unix domain socket with a real peer.
//!
//! The conversation itself is app-core's (`tests/ipc_protocol.rs`, over an
//! in-memory pipe). What only this tier can show is the part that is about the
//! *file*: that a socket appears where the profile is, with the mode the design
//! claims, that a stale one from a process that died is replaced rather than
//! refused, that a same-user peer is served end to end, and that closing takes
//! the door away.
//!
//! Not testable here, and honestly: the refusal of a **foreign** uid. Making a
//! connection from another user needs another user, which a test running as one
//! unprivileged account cannot conjure. The decision itself
//! (`ipc::admit`) is pure and unit-tested in the module — including the case
//! where the kernel will not name the peer, which fails closed — and this
//! suite covers the half that says a same-user peer is admitted at all.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;

use eidola_app_core::AppCore;
use eidola_app_core::ipc::{
    Call, HelloResult, PROTOCOL_VERSION, Request, ResponseBody, encode_line,
};
use eidola_gui::ipc;

/// A real `AppCore` over tempdirs. Nothing here reaches the network.
fn core() -> (Arc<AppCore>, tempfile::TempDir) {
    // Idempotent crypto-provider install (mirrors what `AppCore::new` needs).
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let dir = tempfile::tempdir().expect("tempdir");
    let config_dir = dir.path().to_path_buf();
    let data_dir = dir.path().join("data");
    let core = AppCore::new(config_dir, data_dir).expect("open core");
    (Arc::new(core), dir)
}

/// Say hello over a fresh connection and read the answer back.
fn handshake(path: &std::path::Path) -> HelloResult {
    let mut stream = std::os::unix::net::UnixStream::connect(path).expect("connect");
    stream
        .write_all(&encode_line(&Request::new(1, &Call::Hello)))
        .expect("write");
    stream.flush().expect("flush");
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).expect("read");
    let frame = eidola_app_core::ipc::decode_response(line.trim_end().as_bytes())
        .expect("the app wrote a frame");
    assert_eq!(frame.id, 1);
    match frame.body {
        ResponseBody::End { data } => serde_json::from_value(data).expect("a hello result"),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn the_socket_appears_beside_the_profile_and_answers_the_user_who_owns_it() {
    let (core, _dir) = core();
    let socket = ipc::serve(&core).expect("bind");
    let path = socket.path().to_path_buf();

    assert_eq!(
        path,
        core.data_dir().join(eidola_app_core::ipc::SOCKET_NAME),
        "the socket belongs to the profile that is actually open"
    );
    let mode = std::fs::metadata(&path)
        .expect("metadata")
        .permissions()
        .mode();
    assert_eq!(
        mode & 0o777,
        0o600,
        "the mode is set explicitly, not left to the umask"
    );

    let hello = handshake(&path);
    assert_eq!(hello.protocol, PROTOCOL_VERSION);
    assert_eq!(
        hello.app_version,
        env!("CARGO_PKG_VERSION"),
        "the app answers with its own version"
    );

    socket.close();
}

#[test]
fn a_socket_left_behind_by_a_dead_process_is_replaced() {
    let (core, _dir) = core();
    // Whatever this file is, nothing can be listening on it: this process holds
    // the profile's single-writer lock. That is what makes replacing it safe.
    std::fs::create_dir_all(core.data_dir()).expect("data dir");
    let stale = std::os::unix::net::UnixListener::bind(
        core.data_dir().join(eidola_app_core::ipc::SOCKET_NAME),
    )
    .expect("a stale socket");
    drop(stale);

    let socket = ipc::serve(&core).expect("bind over the stale socket");
    let hello = handshake(socket.path());
    assert_eq!(hello.protocol, PROTOCOL_VERSION);
    socket.close();
}

#[test]
fn more_than_one_caller_can_be_connected_at_once() {
    let (core, _dir) = core();
    let socket = ipc::serve(&core).expect("bind");
    let path = socket.path().to_path_buf();

    // Held open across each other: connections are independent, and a client
    // that is still connected must not be in anyone else's way.
    let first = std::os::unix::net::UnixStream::connect(&path).expect("connect");
    let second = std::os::unix::net::UnixStream::connect(&path).expect("connect");
    assert_eq!(handshake(&path).protocol, PROTOCOL_VERSION);
    drop(first);
    drop(second);
    assert_eq!(
        handshake(&path).protocol,
        PROTOCOL_VERSION,
        "and the door is still open after they leave"
    );

    socket.close();
}

#[test]
fn closing_takes_the_door_away() {
    let (core, _dir) = core();
    let socket = ipc::serve(&core).expect("bind");
    let path = socket.path().to_path_buf();
    assert!(path.exists());

    socket.close();

    assert!(!path.exists(), "a full shutdown removes the socket file");
    assert!(
        std::os::unix::net::UnixStream::connect(&path).is_err(),
        "and there is nothing left to connect to"
    );
}

/// A connection that stays open, so a test can ask it something twice.
fn connect(
    path: &std::path::Path,
) -> (
    std::os::unix::net::UnixStream,
    BufReader<std::os::unix::net::UnixStream>,
) {
    let stream = std::os::unix::net::UnixStream::connect(path).expect("connect");
    // A caller that is never answered must fail rather than hang the suite.
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .expect("read timeout");
    let reader = BufReader::new(stream.try_clone().expect("clone"));
    (stream, reader)
}

/// Ask over an established connection; `None` when the app answered nothing.
fn ask(
    stream: &mut std::os::unix::net::UnixStream,
    reader: &mut BufReader<std::os::unix::net::UnixStream>,
    id: u64,
    call: &Call,
) -> Option<ResponseBody> {
    // A write that fails is itself an answer: the connection is gone.
    stream
        .write_all(&encode_line(&Request::new(id, call)))
        .ok()?;
    stream.flush().ok()?;
    let mut line = String::new();
    if reader.read_line(&mut line).ok()? == 0 {
        return None;
    }
    let frame = eidola_app_core::ipc::decode_response(line.trim_end().as_bytes())
        .expect("the app wrote a frame");
    assert_eq!(frame.id, id);
    Some(frame.body)
}

#[test]
fn closing_ends_the_connections_it_had_already_accepted() {
    // Aborting the accept loop stops the *next* peer. The ones already inside
    // are tasks of their own, and a task nobody holds outlives the listener
    // that spawned it — so a peer connected when the app began quitting could
    // go on dispatching through the whole asynchronous shutdown grace, long
    // enough to start a billed turn moments before the process ends.
    let (core, _dir) = core();
    let socket = ipc::serve(&core).expect("bind");
    let (mut stream, mut reader) = connect(socket.path());

    // Genuinely established: it has completed the handshake on this very
    // connection, so what follows is about a peer the app already knows.
    assert!(
        matches!(
            ask(&mut stream, &mut reader, 1, &Call::Hello),
            Some(ResponseBody::End { .. })
        ),
        "the connection is being served"
    );

    socket.close();

    assert!(
        ask(
            &mut stream,
            &mut reader,
            2,
            &Call::SpacesList {
                include_archived: false
            },
        )
        .is_none(),
        "a closed door went on answering the peers already through it"
    );
}
